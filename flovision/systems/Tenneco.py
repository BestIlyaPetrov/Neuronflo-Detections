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

class TennecoInferenceSystem(InferenceSystem):
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
        self.tracker_id_side_entered = [{} for _ in range(len(self.cams))]
    
    def findCenter(self, minX, minY, maxX, maxY):
        # Will find the center of a detection when given the 
        # min and max for the X and Y coordinates. 
        centerX = round(minX + (maxX - minX)/2)
        centerY = round(minY + (maxY - minY)/2)
        return centerX, centerY

    def tracker_id_side_entered_update(self) -> None:
        # List comprehension to dump old track ids
        # Will dump a track id if it's older than 1 min
        # key = tracker_id
        # value = ["L or R", datetime_obj] 
        if self.tracker_id_side_entered[self.camera_num]:
            self.tracker_id_side_entered[self.camera_num] = {key:value for key, value in self.tracker_id_side_entered[self.camera_num].items() if (datetime.datetime.now() - value[1]) < datetime.timedelta(minutes=1)}

    def Update_Dictionary(self) -> None:
        # Will update the violation_dictionary to contain the
        # most up to date violations. If there's a violation
        # that's over 10 minutes old, that violation will 
        # be deleted from the dictionary.  
        self.violation_dictionary[self.camera_num] = {key: value for key, value in self.violation_dictionary[self.camera_num].items() if len(value)}
        
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
            # violations = [[class_id, detection_index, detection_track_id],
            #               [class_id, detection_index, detection_track_id], ...]


        #if at least one violation is detected, let's record N frames and decide if it was a fluke or not    
        #print(len(self.violations))
        return bool(len(self.violations))

    def annotate_violations(self) -> list:
        """
        NOT TESTED
        """
        # Use this function to annotate the frame's
        # valid violations.   
        violations = self.violation_to_server[self.camera_num]
        least_blurry_indx = least_blurry_image_indx(self.array_for_frames[self.camera_num])
        frame = self.array_for_frames[self.camera_num][least_blurry_indx]
        detections = self.detections_array[self.camera_num][least_blurry_indx]
        # violation = [person_index, soldering_iron_index, camera_index, violation_code, track_id]
        # violations = [violation, violation, ...]
        person_text = 'No Goggles'
        shoes_text = 'Wrong Shoes'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)  # Red color in BGR format
        box_color = (0, 0, 255)
        line_thickness = 2
        font_thickness = 2

        # violations = [[class_id, detection_index, detection_track_id],
        #               [class_id, detection_index, detection_track_id], ...]

        for violation in violations:
            # Each iteration will text annotate a full violation
            # onto the frame. Each annotation will have a [int] 
            # at the end of the text annotation from the frame.
            # This is to indicate which detection is with which
            # that caused a violation 
            class_id = violation[0]
            detection_idx = violation[1]
            Xmin, Ymin, Xmax, Ymax = detections.xyxy[detection_idx]
            positions = [(int(Xmin), int(Ymin)), (int(Xmax), int(Ymax))]
            text_position = (int(Xmin), int(Ymin)-5)
            annotation_text = person_text if class_id == 1 else shoes_text
            # For the text background
            # Finds space required by the text so that we can put a background with that amount of width.
            (w, h), _ = cv2.getTextSize(annotation_text, font, font_scale, font_thickness)
            # Text background
            frame = cv2.rectangle(frame, (Xmin, Ymin - 20), (Xmin + w, Ymin), box_color, -1)
            # Label text
            frame = cv2.putText(frame, annotation_text, text_position, font, font_scale, font_color, font_thickness)
            # Bounding box
            frame = cv2.rectangle(frame, positions[0], positions[1], box_color, line_thickness)

        return frame

    def trigger_action(self) -> None:
        """
        NOT TESTED
        """

        """
        NOTES
        1) Ask Ilya about hard coding the camera's index to be able to control
           which camera is getting consecutive frames
        """
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
                # For this code to work, remote that the the last index of violation must
                # be the tracker id of that set of detections
                if violation[-1] in violating_ids_list:
                    corrected_violations.append(violation)

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
                class_id = violation[0]
                timestamp = datetime.datetime.now()
                violation_code = 0 if class_id == 1 else 1 
                # Meaning that if there's no goggles detected 
                # the violation code will be 0 and if no boots
                # are detected then the code will be 1.
 
                track_id = violation[-1] # self.detections.tracker_id[violation[0]]
                # Creates a violation object to be stored in the 
                self.violation_dictionary[self.camera_num][track_id] = Violation(camera_id=camera_id, class_id=class_id, timestamp=timestamp, violation_code=violation_code)
                # After dict is updated, prep to send to server
                self.violation_to_server[self.camera_num].append(violation)

            # For loop for processing old violations with a track_id
            for violation in old_violations:
                # Check if violation object has record of the same violation 
                # in the last 10 minutes 
                class_id = violation[0]
                violation_code = 0 if class_id == 1 else 1 
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
        rules_broken = ["No goggles were detected." if self.camera_num == 0 else "Wrong shoes detected."]

        data = {
            'num_of_violators': str(len(self.violation_to_server[self.camera_num])),
            'timestamps': timestamp_to_send, # We only need a timestamp
            'rules_broken': str(rules_broken),
            'compliant': "False"
        }

        # send the actual image to the server
        sendImageToServer(self.frame_with_violation, data, IP_address=self.server_IP)

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
        no_boots_class = 3
        
        # Assuming that the model will have goggles/no_goggles
        # and boots/no_boots classes 
        if len(detections) == 0:
            return []
        self.detections = detections
        self.labels = self.detections.class_id
        self.no_goggles_index = [index for index, label in enumerate(self.labels) if label == no_goggles_class]
        self.no_boots_index = [index for index, label in enumerate(self.labels) if label == no_boots_class] 

        return self.process()

    def process(self) -> list:
        # Will run all of the violation detecting methods here
        # and return a list of lists. Each element will hold the
        # the index of the person violating the rule, the index
        # of the item that causes the person to be in violation,
        # the camera index, and then the violation code.

        violations = self.violation_creator()
        # [[class_id, detection_index, detection_track_id],
        #  [class_id, detection_index, detection_track_id], ...]
        #print(violations)
        return violations
    
    def violation_creator(self) -> list:
        violations = []
        no_goggles_class = 1
        no_boots_class = 3
        self.system.tracker_id_side_entered_update()
        # Add the index, class, and track ID to the list of violations
        if self.system.camera_num == 0:
            for no_goggle_index in self.no_goggles_index:
                track_id = self.detections.tracker_id[no_goggle_index]
                violation = [no_goggles_class, no_goggle_index, track_id]

                # If track id is not in the system yet, place it in 
                if track_id not in self.system.tracker_id_side_entered[self.system.camera_num]:
                    minX, minY, maxX, maxY = self.detections.xyxy[no_goggle_index]
                    centerX, centerY = self.system.findCenter(minX=minX, minY=minY, maxX=maxX, maxY=maxY)
                    if float(centerX)/float(self.system.frame_width) < 0.5:
                        self.system.tracker_id_side_entered[self.system.camera_num][track_id] = ["L", datetime.datetime.now()]
                    else:
                        self.system.tracker_id_side_entered[self.system.camera_num][track_id] = ["R", datetime.datetime.now()]
                    
                # If the tracker_id already exists and showed that the detection entered from a side 
                # that indicated someone leaving the facility, then the violation will be skipped 
                if self.system.tracker_id_side_entered[self.system.camera_num][track_id] != None:
                    if self.system.tracker_id_side_entered[self.system.camera_num][track_id][0] == 'L':
                        continue
                
                violations.append(violation)
        else:
            for no_boots_index in self.no_boots_index:
                track_id = self.detections.tracker_id[no_boots_index]
                violation = [no_boots_class, no_boots_index, track_id]
                
                # If track id is not in the system yet, place it in 
                if track_id not in self.system.tracker_id_side_entered[self.system.camera_num]:
                    minX, minY, maxX, maxY = self.detections.xyxy[no_boots_index]
                    centerX, centerY = self.system.findCenter(minX=minX, minY=minY, maxX=maxX, maxY=maxY)
                    if float(centerX)/float(self.system.frame_width) < 0.5:
                        self.system.tracker_id_side_entered[self.system.camera_num][track_id] = ["L", datetime.datetime.now()]
                    else:
                        self.system.tracker_id_side_entered[self.system.camera_num][track_id] = ["R", datetime.datetime.now()]
                # a = [{1:["L", datetime_obj]}, {}, {}]
                # If the tracker_id already exists and showed that the detection entered from a side 
                # that indicated someone leaving the facility, then the violation will be skipped 
                if self.system.tracker_id_side_entered[self.system.camera_num][track_id] != None:
                    if self.system.tracker_id_side_entered[self.system.camera_num][track_id][0] == 'L':
                        continue
                
                violations.append(violation)

        # Now a violation will only occur if there's someone with the wrong shoes or 
        # wearing no goggles detected  
        return violations