
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

from collections import Counter
from math import floor


"""
Model detects the following classes
0: goggles
1: no_goggles
2: soldering
"""
class FrameProcessing():
    # A class for processing detections found in a frame  
    def __init__(self, inference_system) -> None:
        self.detections = None
        self.system = inference_system

    def NewDetections(self, detections):
        # Will be used to update the detections and
        # returns the relevant detection data 
        # relating to violations detected so far 
        if len(detections) == 0:
            return []
        self.detections = detections
        return self.process()

    def process(self) -> list:
        # Will run all of the violation detecting methods here
        # and return a list of lists. Each element will hold the
        # the index of the person violating the rule, the index
        # of the item that causes the person to be in violation,
        # the camera index, and then the violation code. 
        violations = self.distance_rule(detections=self.detections)
        return violations

    def distance_rule(self, detections) -> list:
        # This method will evaluate the detections found
        # in the frame if they're valid violations 
        labels = detections.class_id
        camera_index = 0
        soldering_iron_index = 0
        threshold = 0.25
        frame_violations = []
        # Change the class numbers for the labels to the correct one
        # Change the class numbers for the labels to the correct one
        for label in labels:
            # Iterates through all label to find the active soldering iron
            if label == 2:
                # When active soldering iron is found
                # Find the center point of it 
                minX, minY, maxX, maxY = detections.xyxy[soldering_iron_index]
                centerX, centerY = self.system.findCenter(minX=minX, minY=minY, maxX=maxX, maxY=maxY)
                
                # Start iterating through the detections to find people with no goggles
                person_index = 0
                for label2 in labels:
                    if label2 == 1: # Assuming class 1 is "no_goggles"
                        # First find the center point of the person with no goggles
                        minX2, minY2, maxX2, maxY2 = detections.xyxy[person_index]
                        centerX2, centerY2 = self.findCenter(minX=minX2, minY=minY2, maxX=maxX2, maxY=maxY2)
                        # Second find the distance between the soldering iron and person
                        # with no goggles in the x and y direction 
                        distX = abs(centerX - centerX2)/self.system.frame_width
                        distY = abs(centerY - centerY2)/self.system.frame_height
                        # Third compare the distances to the threshold 
                        if distX < threshold and distY < threshold:
                            # If true, then append the relevant violation information
                            # to the violations list 
                            violation_code = 0
                            violation = [person_index, soldering_iron_index, camera_index, violation_code, detections.track_id[person_index]]
                            frame_violations.append(violation)
                    # Iterate the index for keeping track of people with no goggles
                    person_index = person_index + 1
            # Iterate the index for keeping track of active soldering irons
            soldering_iron_index = soldering_iron_index + 1
        # Return the final frame violations that were valid
        return frame_violations


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
    
    def run(self, iou_thres, agnostic_nms):
        print("Inference successfully launched")
        self.detection_trigger_flag = False
        self.byte_tracker = sv.ByteTrack()
        self.FrameProcessor = FrameProcessing(inference_system=self)
        self.violation_dictionary = [{} for _ in range(len(self.cams))]
        self.detections = []    
        self.camera_num = 0 # the index of the video stream being processed

        while True:
            try:
                self.violation_to_server = []
                # Make the slowest cam be the bottleneck here
                ret, frame = self.cams[0].getFrame()
                #ret2, frame2 = self.cams[1].getFrame()
                #if ret == False or ret2 == False:
                if ret == False:
                    continue

                # Run Inference
                results = self.model(frame)

                # load results into supervision
                self.detections = sv.Detections.from_yolov5(results)
                
                # Apply NMS to remove double detections
                self.detections = self.detections.with_nms(threshold=iou_thres,  class_agnostic=agnostic_nms)

                # Gives all detections ids and will be processed in the next step
                if len(self.detections) > 0:
                    self.trigger_event()
                # Printing out raw detections will output the first one and 
                # printing out the bytetracker implementation will output 
                # the second one:
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

                if self.detection_trigger_flag:
                    self.trigger_action()

                # Display frame
                if self.display:
                    cv2.imshow('ComboCam', frame)

            except Exception as e:
                print('frame unavailable', e)
                traceback.print_exc()

            if cv2.waitKey(1) == ord('q'):
                self.stop()
    
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



        
    def repeat_ids(list_of_track_ids):
        # Flatten the list
        flat_list = [item for sublist in list_of_track_ids for item in sublist]
        
        # Count occurrences
        count = Counter(flat_list)
        
        # Get numbers that appear at least floor(len(a)/2) times
        return [num for num, freq in count.items() if freq >= floor(len(list_of_track_ids)/2)]





    def trigger_event(self) -> bool:
        # Will change the self.detection_trigger_flag to True
        # if a valid violation is detected. 
        if self.detection_trigger_flag[self.camera_num]:
            return False
        # self.detections = self.ByteTracker_implementation(detections=self.detections, byteTracker=self.byte_tracker)

        # violations should be a list of lists
        self.violations = self.FrameProcessor.NewDetections(detections=self.detections)
            # violations = [[person_index, soldering_iron_index, camera_index, violation_code],
            #               [person_index, soldering_iron_index, camera_index, violation_code], ...]
            # tracker_id = detections.tracker_id[detection_info[1]]

        #if at least one violation is detected, let's record N frames and decide if it was a fluke or not    
        return len(self.violations)

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
        if  self.consecutive_frames_cnt[self.camera_num] >= self.num_consecutive_frames:

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

            corrected_violations = []
            for violation in self.violations_array[self.camera_num][0]:
                if violation[-1] in violating_ids_list:
                    corrected_violations.append(violation)


        """
        #1. Check if id of person is inside violations dictionary -> if-statment
        #2. If not, add them and create a new Violation object and add to dictionary -> if-statement
        3. If they are, then check if they've made the same violation in the past -> functionA
            a. If they have made the same violation in the past 10 minutes with the same class_id
               detected, then ignore this violation. If the class_id is different, then it's a valid
               violation.
            b. If they haven't made this violation in the past 10 minutes, then the violation is
               valid.
        4. Move onto next violation
        """
        # Skip if there are no violations
        if len(corrected_violations):

            self.Update_Dictionary()
            # Iterate through all violations initially detected
            # violation = [person_index, soldering_iron_index, camera_index, violation_code]
            new_violations = [violation for violation in violations if violation[0] in self.violation_dictionary[self.camera_num]]
            old_violations = [violation for violation in violations if not violation[0] in self.violation_dictionary[self.camera_num]]
            
            # Makes sure there's 
            # self.violation_dictionary[self.camera_num] = {key: value for key, value in self.violation_dictionary[self.camera_num].items() if len(value)}
            
            # For loop for processing new violations with no track_id 
            for violation in new_violations:
                # If not then add new violation object to the dictionary
                camera_id = self.camera_num
                class_id = self.detections.class_id[violation[1]]
                timestamp = datetime.datetime.now()
                violation_code = 0
                track_id = self.detections.tracker_id[violation[0]]
                # Creates a violation object to be stored in the 
                self.violation_dictionary[self.camera_num][track_id] = Violation(camera_id=camera_id, class_id=class_id, timestamp=timestamp, violation_code=violation_code)
                # After dict is updated, prep to send to server
                self.violation_to_server.append(violation)
                

            # For loop for processing old violations with a track_id
            for violation in old_violations:
                # Check if violation object has record of the same violation 
                # in the last 10 minutes 
                class_id = self.detections.class_id[violation[1]]  
                violation_code = 0
                violation_object = self.violation_dictionary[self.camera_num][violation[0]]
                if violation_object.Check_Code(violation_code, class_id):
                    # If true, violation already exists and is not valid
                    continue
                else:
                    # If false, then that means that violation is valid and
                    # should be added to the list of violations to be sent
                    # to the server 
                    self.violation_to_server.append(violation)
                    # Add code to modify the violation object
                    violation_object.Add_Code(violation_code=violation_code, class_id=class_id)
         

            # Pick the least blurry image to send to the server (we assume images don't vary that much within a small enough del_t or in this case N = self.num_consecutive_frames)
            least_blurry_indx = least_blurry_image_indx(self.array_for_frames[self.camera_num])

            # PLACE LOGIC FOR ANNOTATIONS HERE!
            # PLACE LOGIC FOR ANNOTATIONS HERE! 
            # PLACE LOGIC FOR ANNOTATIONS HERE!
            # PLACE LOGIC FOR ANNOTATIONS HERE!
            # PLACE LOGIC FOR ANNOTATIONS HERE!

            #  Compliance Logic
            img_to_send = self.array_for_frames[self.camera_num][least_blurry_indx]

            # Pack the data for the server to process - TODO: figure out what data we are sending to the server - what can we gather?
            # FIX THIS DATA THAT IS BEING SENT TO THE SERVER
            
            # What we want to be sent to the server:
            # GOOD 1) Image
            # GOOD 2) Number of people breaking the rules
            # GOOD 3) Time of violation
            # GOOD 4) Name of rule broken
            
            # violation = [person_index, soldering_iron_index, camera_index, violation_code, track_id] inside self.violation_to_server
            # self.violation_to_server = [[violation, violation, ...], [violation, violation, ...], ...]
            # self.violation_dictionary = [{person_track_id:violation_object, ...}, {person_track_id:violation_object, ...}, ...]
            timestamps = [self.violation_dictionary[self.camera_num][violation[-1]].Get_Timestamp(violation[-2]).strftime('%Y-%m-%dT%H:%M:%SZ') for violation in self.violation_to_server]
            rules_broken = ["Too close to active soldering iron." for violation in self.violation_to_server[self.camera_num] if violation[-2] == 0]
            
            data = {
                'num_of_violators': str(len(self.violation_to_server[self.camera_num])),
                'timestamps': ','.join(timestamps),
                'rules_broken': str(rules_broken),
                'compliant': "False"
            }

            # send the actual image to the server
            sendImageToServer(img_to_send, data, IP_address=self.server_IP)



            # Save the image locally for further model retraining
            if self.save:
                self.save_frames(self.array_for_frames[self.camera_num])

        # Reset the arrays for the data and the images, since we just sent it to the server
        self.detections_array[self.camera_num] = []
        self.array_for_frames[self.camera_num] = []
