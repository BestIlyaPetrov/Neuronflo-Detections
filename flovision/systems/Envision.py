from threading import Thread
import cv2
import time
import numpy as np
import torch, torchvision
import json
import datetime
import time

import supervision as sv
import traceback
import collections
from pathlib import Path

from ..bbox_gui import create_bounding_boxes, load_bounding_boxes
from ..video import draw_border, region_dimensions, vStream, least_blurry_image_indx, get_device_indices
from ..comms import sendImageToServer
from ..utils import get_highest_index, findLocalServer
from ..jetson import Jetson
from ..face_id import face_recog
from ..inference import InferenceSystem, Violation
from ..notifications import teleBot

from collections import Counter
from math import floor
import os


"""
Model detects the following classes
0-goggles
1-no_goggles
2-soldering
3-hand

COCO dataset
0 - person
27 - tie
39 - bottle
42 - fork 
43 - knife
44 - spoon 
45 - bowl
56 - chair
57 - couch
59 - bed
60 - dining table
62 - tv
63 - laptop
64 - mouse
65 - remote
66 - keyboard
67 - cell phone
69 - oven
70 - toaster
71 - sink
72 - refridgerator
73 - book
76 - scissors
79 - toothbrush
"""

class EnvisionInferenceSystem(InferenceSystem):
    def __init__(self, *args, **kwargs) -> None:

        """
        self.frame_width = video_res[0]
        self.frame_height = video_res[1]
        self.frame_size = (self.frame_width, self.frame_height)
        self.border_thickness = border_thickness
        self.display = display
        self.save = save
        self.annotate = annotate
        """

        # If need to overwrite a particular argument do the following. 
        # Let's say need to overwrite the 'model_directory' argument
        # kwargs['model_directory'] = 'my_new_directory_path'

        super().__init__(*args, **kwargs)

        # List of lists, beacause for each camera feed, we have a list of violations we save to choose the prevalent one as jitter reduction technique (aka goggle, no_goggle, goggle -> goggle )
        self.violations_array = [[] for _ in range(len(self.cams))]
        self.violations_track_ids_array = [[] for _ in range(len(self.cams))]
        self.FrameProcessor = FrameProcessing(inference_system=self)
        self.violation_dictionary = [{} for _ in range(len(self.cams))]
        self.violation_to_server = [[] for _ in range(len(self.cams))]
        # API_token = '6323749554:AAEAA_qF1dDE-UWlTr9nxlqlj_pmZbNOqSY'
        # self.telegram_bot = teleBot(API_TOKEN=API_token, name='UCSD Envision Inference')
        '''
                Detections(xyxy=array([[     816.08,         243,      905.23,      368.74],
                                       [     81.858,       243.4,      168.83,      364.49],
                                       [     414.21,      267.22,       498.5,      372.73]], 
                                      dtype=float32), 
                           mask=None, 
                           confidence=array([    0.84752,     0.76186,     0.49337], dtype=float32), 
                           class_id=array([1, 2, 0]), 
                           tracker_id=None)
                NEW DETECTIONS:  Detections(xyxy=array([[     77.585,      243.67,      168.21,      364.91],
                                                        [      808.9,      242.57,      904.81,      368.67],
                                                        [     415.05,      267.64,      497.65,       373.7]], 
                                                        dtype=float32), 
                                            mask=None, 
                                            confidence=array([    0.76186,     0.84752,     0.49337], dtype=float32), 
                                            class_id=array([1, 2, 0]), 
                                            tracker_id=array([19, 20, 21]))

            # violations = [[person_index, soldering_iron_index, camera_index, violation_code, detections.track_id[person_index]],
            #               [person_index, soldering_iron_index, camera_index, violation_code, detections.track_id[person_index]], ...]
        '''

    def Update_Dictionary(self) -> None:
        # Will update the violation_dictionary to contain the
        # most up to date violations. If there's a violation
        # that's over 10 minutes old, that violation will 
        # be deleted from the dictionary.  
        self.violation_dictionary[self.camera_num] = {key: value for key, value in self.violation_dictionary[self.camera_num].items() if len(value)}

    def findCenter(self, minX, minY, maxX, maxY):
        # Will find the center of a detection when given the 
        # min and max for the X and Y coordinates. 
        centerX = round(minX + (maxX - minX)/2)
        centerY = round(minY + (maxY - minY)/2)
        return centerX, centerY
        
    def repeat_ids(self,list_of_track_ids):
        # Flatten the list
        flat_list = [item for sublist in list_of_track_ids for item in sublist]
        flat_list = [tuple(x) for x in flat_list]

        # Count occurrences
        count = Counter(flat_list)
        
        # Get numbers that appear at least floor(len(a)/2) times
        return [num for num, freq in count.items() if freq >= floor(len(list_of_track_ids)/2)]

    def trigger_event(self) -> bool:
        # Will change the self.detection_trigger_flag to True
        # if a valid violation is detected. 
        if self.detection_trigger_flag[self.camera_num]:
            return False

        # violations should be a list of lists
        self.violations = self.FrameProcessor.NewDetections(detections=self.detections)
            # violations = [[person_index, soldering_iron_index, camera_index, violation_code, track_id],
            #               [person_index, soldering_iron_index, camera_index, violation_code, track_id], ...]
            # tracker_id = detections.tracker_id[detection_info[1]]

        #if at least one violation is detected, let's record N frames and decide if it was a fluke or not    
        #print(len(self.violations))
        return bool(len(self.violations))

    def annotate_violations(self) -> list:
        # Use this function to annotate the frame's
        # valid violations.   
        violations = self.violation_to_server[self.camera_num]
        least_blurry_indx = least_blurry_image_indx(self.array_for_frames[self.camera_num])
        frame = self.array_for_frames[self.camera_num][least_blurry_indx]
        detections = self.detections_array[self.camera_num][least_blurry_indx]
        # violation = [person_index, soldering_iron_index, camera_index, violation_code, track_id]
        # violations = [violation, violation, ...]
        person_text = 'No Goggles'
        soldering_text = 'Active Soldering Iron'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 0, 255)  # Red color in BGR format
        line_thickness = 2
        for violation in violations:
            # Each iteration will text annotate a full violation
            # onto the frame. Each annotation will have a [int] 
            # at the end of the text annotation from the frame.
            # This is to indicate which detection is with which
            # that caused a violation 

            # Person annotation info-variables
            person_idx = violation[0]
            person_Xmin, person_Ymin, person_Xmax, person_Ymax = detections.xyxy[person_idx]
            person_position = (int(person_Xmin), int(person_Ymin))
            person_position2 = (int(person_Xmax), int(person_Ymax))

            # Solder annotation info-variables
            solder_idx = violation[1]
            solder_Xmin, solder_Ymin, solder_Xmax, solder_Ymax = detections.xyxy[solder_idx]
            solder_position = (int(solder_Xmin), int(solder_Ymin))
            solder_position2 = (int(solder_Xmax), int(solder_Ymax))
            #print(f"person_position = {person_position}")
            #print(f"solder_position = {solder_position}")

            # Add text annotations to the frame
            solder_annotation = f"{soldering_text}"
            person_annotation = f"{person_text}"

            # Frame manipulation
                # Soldering Irons
            frame = cv2.putText(frame, solder_annotation, solder_position, font, font_scale, font_color, line_thickness)
            frame = cv2.rectangle(frame, solder_position, solder_position2, font_color, line_thickness)
                # People With No Goggles
            frame = cv2.putText(frame, person_annotation, person_position, font, font_scale, font_color, line_thickness)
            frame = cv2.rectangle(frame, person_position, person_position2, font_color, line_thickness)
        return frame

    def trigger_action(self) -> None:
        # Count how many images we already took
        self.consecutive_frames_cnt[self.camera_num] += 1
        
        # The following 3 arrays are appened in parallel

        # Append detections from self.camera_num source to the array for detection jitter reduction
        self.detections_array[self.camera_num].append(self.detections)

        if self.consecutive_frames_cnt[self.camera_num] > 1: #if cnt==1, we just saved the self.violations in the trigger_event(), no need to do it again
            self.violations = self.FrameProcessor.NewDetections(detections=self.detections)

        # Append violations found in the detections above
        self.violations_array[self.camera_num].append(self.violations)

        # Append an image from self.camera_num source to the array for least blurry choice
        self.array_for_frames[self.camera_num].append(self.captures[self.camera_num])

        # After N consecutive frames collected, check what was the most common detection for each object
        corrected_violations = []
        least_blurry_indx = None

        if  self.consecutive_frames_cnt[self.camera_num] >= self.num_consecutive_frames:
            #print("START HERE")
            # Reset the consecutive frames count
            self.consecutive_frames_cnt[self.camera_num] = 0

            # Reset the trigger flag as well so we can have another trigger action
            self.detection_trigger_flag[self.camera_num] = False

            # for each frame check Detection.tracker_id[violation[0]], where violation[0] is the human's tracker_id
            # maybe lets's first create a list of all track ids in violation 

            for violations, detection in zip(self.violations_array[self.camera_num], self.detections_array[self.camera_num]):
                # violation[-1] <- person's track_id 
                track_ids =[]
                for violation in violations:
                    track_ids.append(violation[-1])

                self.violations_track_ids_array[self.camera_num].append(track_ids)
        
            # Now figure out which track_ids repeat from frame to frame 
            # The threshold is that each track_id must happen more than floor(N/2) times
            violating_ids_list = self.repeat_ids(self.violations_track_ids_array)
            
            # Finds the least blurry image's index
            least_blurry_indx = least_blurry_image_indx(self.array_for_frames[self.camera_num])
            
            # This records the union of track_ids of the least blurry image and which 
            # ids are followed in multiple frames 
            for violation in self.violations_array[self.camera_num][least_blurry_indx]:
                if violation[-1] in violating_ids_list:
                    corrected_violations.append(violation)
        #print(f"corrected_violations: {corrected_violations}")
        # Skip if there are no violations
        if len(corrected_violations):
            # Pick the least blurry image to send to the server
            # (we assume images don't vary that much within a 
            # small enough del_t or in this case N = self.num_consecutive_frames)

            # Finds the least blurry image's index
            #least_blurry_indx = least_blurry_image_indx(self.array_for_frames[self.camera_num])
            self.Update_Dictionary()

            # Iterate through all violations initially detected
            # violation = [person_index, soldering_iron_index, camera_index, violation_code, track_id]
            # Problem was that the violation's person_idx was being checked if it was inside the 
            # violation_dictionary when we should be searching for the track id of the person in
            # the violation 
            old_violations = [violation for violation in corrected_violations if violation[-1] in self.violation_dictionary[self.camera_num]]
            new_violations = [violation for violation in corrected_violations if not violation[-1] in self.violation_dictionary[self.camera_num]]

            # For loop for processing new violations with no track_id 
            for violation in new_violations:
                # If not then add new violation object to the dictionary
                camera_id = self.camera_num
                class_id = 3
                timestamp = datetime.datetime.now()
                violation_code = 0
                track_id = violation[-1] # self.detections.tracker_id[violation[0]]
                # Creates a violation object to be stored in the 
                self.violation_dictionary[self.camera_num][track_id] = Violation(camera_id=camera_id, class_id=class_id, timestamp=timestamp, violation_code=violation_code)
                # After dict is updated, prep to send to server
                self.violation_to_server[self.camera_num].append(violation)

            # For loop for processing old violations with a track_id
            for violation in old_violations:
                # Check if violation object has record of the same violation 
                # in the last 10 minutes 
                class_id = 3  
                violation_code = 0
                violation_object = self.violation_dictionary[self.camera_num][violation[-1]]
                if violation_object.Check_Code(violation_code, class_id):
                    # If true, violation already exists and is not valid
                    continue
                else:
                    # If false, then that means that violation is valid and
                    # should be added to the list of violations to be sent
                    # to the server 
                    self.violation_to_server[self.camera_num].append(violation)
                    # Add code to modify the violation object
                    violation_object.Add_Code(violation_code=violation_code, class_id=class_id)

            if len(self.violation_to_server[self.camera_num]):
                self.system_send(least_blurry_indx=least_blurry_indx)
            # Empty the list to be sent to the server after sending 
            self.violation_to_server[self.camera_num] = []
    
    def system_send(self, least_blurry_indx):
        # Define the frame with the least blurry index
        frame = self.array_for_frames[self.camera_num][least_blurry_indx]
        
        # Save the image locally for further model retraining
        if self.save:
            self.save_frames(frame, self.camera_num)
        
        # Annotate the violations
        self.frame_with_violation = self.annotate_violations()

        #  Compliance Logic
        # img_to_send = frame

        timestamp_to_send = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        rules_broken = ["Too close to active soldering iron." for violation in self.violation_to_server[self.camera_num] if violation[-2] == 0]

        data = {
            'zone': f"Envision_table_{self.camera_num}",
            'num_of_violators': str(len(self.violation_to_server[self.camera_num])),
            'timestamps': timestamp_to_send, # We only need a timestamp
            'rules_broken': str(rules_broken),
            'compliant': "False"
        }
        
        link = 'api/violation_update'
        # send the actual image to the server
        sendImageToServer(self.frame_with_violation, data, self.server_IP, link)
            

    def create_file_path(self, frame):
        # Ensure the home directory path is correct for your system
        home_directory = os.path.expanduser("~")
        file_path = os.path.join(home_directory, "captured_frame.jpg")

        # Convert the frame to a JPG image and save it
        cv2.imwrite(file_path, frame)

        return file_path
    
    def Telegram_Notification_Implementation(self, frame) -> None:
        # Telegram Bot Integration
        # First we create the file path for the frame
        file_path = self.create_file_path(frame=frame)
        # Second we send the image to the group chat. Status is used for debugging purposes 
        status = self.telegram_bot.teleImage(file_path=file_path) 
        if len(self.violation_to_server[self.camera_num]) > 1:
            message = f"We have found {len(self.violation_to_server[self.camera_num])} violations!"
        else:
            message = f"We have found {len(self.violation_to_server[self.camera_num])} violation!"
        # Third we send the message describing how many violations were found in the frame sent
        self.telegram_bot.teleMessage(message=message) 
        # Lastly we delete the saved frame from that file pathway
        os.remove(file_path)

class FrameProcessing():
    # A class for processing detections found in a frame  
    def __init__(self, inference_system) -> None:
        self.detections = None
        self.system = inference_system

    def NewDetections(self, detections):
        # Will be used to update the detections and
        # returns the relevant detection data 
        # relating to violations detected so far 
        no_goggles_class = 1
        solder_class = 2
        hand_class = 3
        # no_goggles_class = 65
        # solder_class = 66
        # hand_class = 76
        # This will now activate when a person is 
        # holding a knife and phone in the same hand 
        if len(detections) == 0:
            return []
        self.detections = detections
        self.labels = self.detections.class_id
        self.no_goggles = [index for index, label in enumerate(self.labels) if label == no_goggles_class]
        self.solder_labels = [index for index, label in enumerate(self.labels) if label == solder_class]
        self.hand_labels = [index for index, label in enumerate(self.labels) if label == hand_class]
        # [person_index, soldering_iron_index, camera_index, violation_code, track_id]
        return self.process()

    def process(self) -> list:
        # Will run all of the violation detecting methods here
        # and return a list of lists. Each element will hold the
        # the index of the person violating the rule, the index
        # of the item that causes the person to be in violation,
        # the camera index, and then the violation code.
        if len(self.hand_labels) == 0 or (not self.solder_in_hand()):
            # If there's no hands being detected then it's
            # reasonable to assume that there's no one there
            # to create a violation. If there's no soldering irons
            # inside a hand in frame, then similarly there's need
            # to look for a violation. 
            return []

        violations = self.distance_rule()
        #print(violations)
        return violations

    def solder_in_hand(self) -> bool:
        # The purpose of this function is to edit the 
        # soldering label list to only have soldering
        # irons that are being hand held. 
        threshold = 0.5
        condition = []
        for solder_index in self.solder_labels:
            # Iterate through each soldering iron and look
            # to see if there's a hand close enough to it.
            # If there's a hand close enough, then that's 
            # a valid soldering iron to run violation
            # detection. 
            minX, minY, maxX, maxY = self.detections.xyxy[solder_index]
            centerX, centerY = self.system.findCenter(minX=minX, minY=minY, maxX=maxX, maxY=maxY)
            for hand_index in self.hand_labels:
                # Find the distance between the centers of 
                # the hand and soldering iron detections.
                # If the distance is small enough, append
                # True to condition and False if not. 
                minX, minY, maxX, maxY = self.detections.xyxy[hand_index]
                centerX2, centerY2 = self.system.findCenter(minX=minX, minY=minY, maxX=maxX, maxY=maxY)
                distX = float(abs(centerX - centerX2))/float(self.system.frame_width)
                distY = float(abs(centerY - centerY2))/float(self.system.frame_height)
                #print(f"distX: {distX} \ndistY: {distY}")
                if distX < threshold and distY < threshold:
                    condition.append(True)
                else:
                    condition.append(False)
        self.solder_labels = [label_index for label_index, cond in zip(self.solder_labels, condition) if cond]
        #print(self.solder_labels)
        #print(bool(len(self.solder_labels)))
        return bool(len(self.solder_labels))

    def distance_rule(self) -> list:
        # This method will evaluate the detections found
        # in the frame if they're valid violations 

        threshold = 0.5
        frame_violations = []
        violation_code = 0
        camera_num = self.system.camera_num
        for solder_index in self.solder_labels:
            minX, minY, maxX, maxY = self.detections.xyxy[solder_index]
            centerX, centerY = self.system.findCenter(minX=minX, minY=minY, maxX=maxX, maxY=maxY)
            for no_goggles_index in self.no_goggles:
                minX, minY, maxX, maxY = self.detections.xyxy[no_goggles_index]
                centerX2, centerY2 = self.system.findCenter(minX=minX, minY=minY, maxX=maxX, maxY=maxY)
                distX = float(abs(centerX - centerX2))/float(self.system.frame_width)
                distY = float(abs(centerY - centerY2))/float(self.system.frame_height)
                #print(f"distX: {distX} \ndistY: {distY}")
                if distX < threshold and distY < threshold:

                    track_id = self.detections.tracker_id[no_goggles_index]
                    violation = [no_goggles_index, solder_index, camera_num, violation_code, track_id]
                    frame_violations.append(violation)
        return frame_violations

