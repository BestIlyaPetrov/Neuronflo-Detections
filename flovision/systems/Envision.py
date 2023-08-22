
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
            if label == 0:
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
                        distX = abs(centerX - centerX2)/self.system.frame_size[0]
                        distY = abs(centerY - centerY2)/self.system.frame_size[1]
                        # Third compare the distances to the threshold 
                        if distX < threshold and distY < threshold:
                            # If true, then append the relevant violation information
                            # to the violations list 
                            violation_code = 0
                            violation = [person_index, soldering_iron_index, camera_index, violation_code]
                            frame_violations.append(violation)
                    # Iterate the index for keeping track of people with no goggles
                    person_index = person_index + 1
            # Iterate the index for keeping track of active soldering irons
            soldering_iron_index = soldering_iron_index + 1
        # Return the final frame violations that were valid
        return frame_violations


class EnvisionInferenceSystem(InferenceSystem):
    def __init__(self, *args, **kwargs) -> None:

        # If need to overwrite a particular argument do the following. 
        # Let's say need to overwrite the 'model_directory' argument
        # kwargs['model_directory'] = 'my_new_directory_path'

        super().__init__(*args, **kwargs)
        self.zone_count = [0 for i in range(len(self.cams))]
        self.consecutive_frames_cnt = [0 for i in range(len(self.cams))]
    
    def run(self, iou_thres, agnostic_nms):
        print("Inference successfully launched")
        self.detection_trigger_flag = False
        byte_tracker = sv.ByteTrack()
        FrameProcessor = FrameProcessing(inference_system=self)
        self.violation_dictionary = {}

        self.camera_num = 0 # the index of the vide stream being processed

        while True:
            try:
                violation_to_server = []
                # Make the slowest cam be the bottleneck here
                ret, frame = self.cams[0].getFrame()
                #ret2, frame2 = self.cams[1].getFrame()
                #if ret == False or ret2 == False:
                if ret == False:
                    continue

                # Run Inference
                results = self.model(frame)

                # load results into supervision
                detections = sv.Detections.from_yolov5(results)
                
                # Apply NMS to remove double detections
                detections = detections.with_nms(threshold=iou_thres,  class_agnostic=agnostic_nms)

                # Gives all detections ids and will be processed in the next step
                if len(detections) > 0:
                    detections = self.ByteTracker_implementation(detections=detections, byteTracker=byte_tracker)

                ### ADDED EVERYTHING STARTING FROM HERE TO TRIGGER ACTION ###
                
                # violations should be a list of lists
                violations = FrameProcessor.NewDetections(detections=detections)
                    # violations = [[person_index, soldering_iron_index, camera_index, violation_code],
                    #               [person_index, soldering_iron_index, camera_index, violation_code], ...]
                    # tracker_id = detections.tracker_id[detection_info[1]]
                
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
                if len(violations):
                    # Iterate through all violations initially detected
                    # violation = [person_index, soldering_iron_index, camera_index, violation_code]
                    new_violations = [violation for violation in violations if violation[0] in self.violation_dictionary]
                    old_violations = [violation for violation in violations if not violation[0] in self.violation_dictionary]
                    
                    # Makes sure there's 
                    self.violation_dictionary = {key: value for key, value in self.violation_dictionary.items() if len(value)}
                    
                    # For loop for processing new violations with no track_id 
                    for violation in new_violations:
                        # If not then add new violation object to the dictionary
                        camera_id = 0
                        class_id = detections.class_id[violation[1]]
                        timestamp = datetime.datetime.now()
                        violation_code = 0
                        track_id = detections.tracker_id[violation[0]]
                        # Creates a violation object to be stored in the 
                        self.violation_dictionary[track_id] = Violation(camera_id=camera_id, class_id=class_id, timestamp=timestamp, violation_code=violation_code)
                        violation_to_server.append(violation)

                    # For loop for processing old violations with a track_id
                    for violation in old_violations:
                        # Check if violation object has record of the same violation 
                        # in the last 10 minutes 
                        class_id = detections.class_id[violation[1]]  
                        violation_code = 0
                        violation_object = self.violation_dictionary[violation[0]]
                        if violation_object.Check_Code(violation_code, class_id):
                            # If true, violation already exists and is not valid
                            continue
                        else:
                            # If false, then that means that violation is valid and
                            # should be added to the list of violations to be sent
                            # to the server 
                            violation_to_server.append(violation)
                            # Add code to modify the violation object
                            violation_object.Add_Code(violation_code=violation_code, class_id=class_id)

                ### ADDED EVERYTHING ENDING HERE TO TRIGGER ACTION ###
                ### TODO: ENSURE trigger_event AND trigger_action ARE CORRECT AND TEST WITH JETSON ###
                ### THEN WE TEST AT UCSD ####
                
                # Check if the distance metric has been violated
                # self.detection_trigger_flag, detection_info = self.trigger_event(detections=detections)
                # The detection_info variable will hold a tuple that
                # will be the indices for the objects in violation
                
                
                # Printing out detections will output this:
                '''
                Detections(xyxy=array([[     816.08,         243,      905.23,      368.74],
                                       [     81.858,       243.4,      168.83,      364.49],
                                       [     414.21,      267.22,       498.5,      372.73]], 
                                      dtype=float32), 
                           mask=None, 
                           confidence=array([    0.84752,     0.76186,     0.49337], dtype=float32), 
                           class_id=array([1, 1, 0]), 
                           tracker_id=None)
                NEW DETECTIONS:  Detections(xyxy=array([[     77.585,      243.67,      168.21,      364.91],
                                                        [      808.9,      242.57,      904.81,      368.67],
                                                        [     415.05,      267.64,      497.65,       373.7]], 
                                                        dtype=float32), 
                                            mask=None, 
                                            confidence=array([    0.76186,     0.84752,     0.49337], dtype=float32), 
                                            class_id=array([1, 1, 0]), 
                                            tracker_id=array([19, 20, 21]))
                '''



                #######################################################################
                # Place logic for detecting if the time passed for an ID's violation  #
                # is under 10 mins. If so, then no violation has occurred. However,   #
                # if it's been over 10 mins. Make sure that the function will take    #
                # in the tracker id.                                                  #
                #######################################################################
                # is under 10 mins. If so, then no violation has occurred. However,   #
                # if it's been over 10 mins. Make sure that the function will take    #
                # in the tracker id.                                                  #
                #######################################################################


                # Display frame
                if self.display:
                    cv2.imshow('ComboCam', frame)

            except Exception as e:
                print('frame unavailable', e)
                traceback.print_exc()

            if cv2.waitKey(1) == ord('q'):
                self.stop()
    
    def Update_Dictionary(self):
        self.violation_dictionary = {key: value for key, value in self.violation_dictionary.items() if len(value)}

        pass

    def findCenter(self, minX, minY, maxX, maxY):
        centerX = round(minX + (maxX - minX)/2)
        centerY = round(minY + (maxY - minY)/2)
        return centerX, centerY

    def trigger_event(self) -> bool:
                 # Trigger event for envision
        # Variables in the for loop to be processed
        labels = self.detections.class_id
        cnt = 0
        violation_detected = False
        threshold = 0.3
        # Change the class numbers for the labels to the correct one
        for label in labels:
            if label == 2: # Assuming class 2 is "soldering"
                minX, minY, maxX, maxY = detections.xyxy[cnt]
                centerX, centerY = self.findCenter(minX=minX, minY=minY, maxX=maxX, maxY=maxY)
                cnt2 = 0
                for label2 in labels:
                    if label2 == 1: # Assuming class 1 is "no_goggles"
                        minX2, minY2, maxX2, maxY2 = detections.xyxy[cnt2]
                        centerX2, centerY2 = self.findCenter(minX=minX2, minY=minY2, maxX=maxX2, maxY=maxY2)
                        distX = abs(centerX - centerX2)/self.frame_size[0]
                        distY = abs(centerY - centerY2)/self.frame_size[1]
                        if distX < threshold and distY < threshold:
                            return True, (cnt, cnt2)
                    cnt2 = cnt2 + 1
            if violation_detected:
                break
            cnt = cnt + 1
        return False
        # # Trigger event for envision
        # return self.zones[0].current_count >= 1
    

    def trigger_action(self) -> None:
        # violations should be a list of lists
        violations = FrameProcessor.NewDetections(detections=self.detections)
            # violations = [[person_index, soldering_iron_index, camera_index, violation_code],
            #               [person_index, soldering_iron_index, camera_index, violation_code], ...]
            # tracker_id = detections.tracker_id[detection_info[1]]
        
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
        if len(violations):
            # Iterate through all violations initially detected
            # violation = [person_index, soldering_iron_index, camera_index, violation_code]
            new_violations = [violation for violation in violations if violation[0] in self.violation_dictionary]
            old_violations = [violation for violation in violations if not violation[0] in self.violation_dictionary]
            
            # Makes sure there's 
            self.violation_dictionary = {key: value for key, value in self.violation_dictionary.items() if len(value)}
            
            # For loop for processing new violations with no track_id 
            for violation in new_violations:
                # If not then add new violation object to the dictionary
                camera_id = 0
                class_id = detections.class_id[violation[1]]
                timestamp = datetime.datetime.now()
                violation_code = 0
                track_id = detections.tracker_id[violation[0]]
                # Creates a violation object to be stored in the 
                self.violation_dictionary[track_id] = Violation(camera_id=camera_id, class_id=class_id, timestamp=timestamp, violation_code=violation_code)
                violation_to_server.append(violation)

            # For loop for processing old violations with a track_id
            for violation in old_violations:
                # Check if violation object has record of the same violation 
                # in the last 10 minutes 
                class_id = detections.class_id[violation[1]]  
                violation_code = 0
                violation_object = self.violation_dictionary[violation[0]]
                if violation_object.Check_Code(violation_code, class_id):
                    # If true, violation already exists and is not valid
                    continue
                else:
                    # If false, then that means that violation is valid and
                    # should be added to the list of violations to be sent
                    # to the server 
                    violation_to_server.append(violation)
                    # Add code to modify the violation object
                    violation_object.Add_Code(violation_code=violation_code, class_id=class_id)

        if self.save:
            self.save_frames(self.captures)



