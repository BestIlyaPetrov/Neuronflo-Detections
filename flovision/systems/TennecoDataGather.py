
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
from ..video import draw_border, region_dimensions, vStream, least_blurry_image_indx, get_camera_src
from ..comms import sendImageToServer
from ..utils import get_highest_index, findLocalServer
from ..jetson import Jetson
from ..face_id import face_recog
from ..inference import InferenceSystem, Violation

from collections import Counter
from math import floor

"""
Model detects the following classes
0-goggles
1-no_goggles
2-soldering
3-hand
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
        self.labels = self.detections.class_id
        self.no_goggles = [index for index, label in enumerate(self.labels) if label == 1]
        self.goggles = [index for index, label in enumerate(self.labels) if label == 0]
        if len(self.goggles) == 0 and len(self.no_goggles) == 0:
            return []
        return self.process()

    def process(self) -> list:
        if len(self.goggles):
            # Assumes there's at least one goggle detection
            thelist = [[0, goggle_idx, self.detections.tracker_id[goggle_idx]] for goggle_idx in self.goggles]
            if len(self.no_goggles):
                # Assumes there's at least one no_goggle detection
                for no_goggle_idx in self.no_goggles:
                    thelist.append([1, no_goggle_idx, self.detections.tracker_id[no_goggle_idx]])
        elif len(self.no_goggles):
            # Assumes there's at least one no_goggle detection and zero goggle detections
            thelist = [[1, no_goggle_idx] for no_goggle_idx in self.no_goggles]
        # thelist = [[class_id, class_id's index], [class_id, class_id's index], ...]
        return thelist

class TennecoImageCapture(InferenceSystem):
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
        self.FrameProcessor = FrameProcessing(inference_system=self)
        self.seen_track_ids = []
        # List of lists, beacause for each camera feed, we have a list of violations we save to choose the prevalent one as jitter reduction technique (aka goggle, no_goggle, goggle -> goggle )

    def trigger_event(self) -> bool:
        self.violations = self.FrameProcessor.NewDetections(detections=self.detections)
        
        # self.violations = [[class_id, detection_idx, tracker_id], [class_id, detection_idx, tracker_id], ...] 
        # violation = [class_id, detection_idx, tracker_id]
        # Add method for updating the self.seen_track_ids list
        self.update_seen_ids()
        cond = []
        timestamp = datetime.datetime.now()
        # Will iterate through all found detections and make sure no 
        # tracker_id for the detections is repeated. This will make 
        # sure that we get a more diverse dataset.  
        for violation in self.violations:
            # if tracker_id inside self.seen_track_ids
            if violation[2] in self.seen_track_ids:
                # Delete violation from self.violations
                cond.append(False)
            # else
            else:
                # Keep the violation and add the track_id to self.seen_track_ids with a timestamp
                self.seen_track_ids.append(violation[2], timestamp)
                cond.append(True)
        self.violations = [violation for violation, condition in zip(self.violations, cond) if condition]
        return bool(len(self.violations))

    def trigger_action(self) -> None:
        # Count how many images we already took
        self.consecutive_frames_cnt[self.camera_num] += 1

        # Append an detections from self.camera_num source to the array for detection jitter reduction
        self.detections_array[self.camera_num].append(self.item_detections)

        # Append an image from self.camera_num source to the array for least blurry choice
        self.array_for_frames[self.camera_num].append(self.captures[self.camera_num])

        # After N consecutive frames collected, check what was the most common detection for each object
        if self.consecutive_frames_cnt[self.camera_num] >= self.num_consecutive_frames:

            # Reset the consecutive frames count
            self.consecutive_frames_cnt[self.camera_num] = 0

            # Reset the trigger flag as well so we can have another trigger action
            self.detection_trigger_flag[self.camera_num] = False
                
            # Pick the least blurry image to send to the server (we assume images don't vary that much within a small enough del_t or in this case N = self.num_consecutive_frames)
            #least_blurry_indx = least_blurry_image_indx(self.array_for_frames[self.camera_num])

            # Save the image locally for further model retraining
            self.save_frames()

            # Reset the arrays for the data and the images, since we just sent it to the server
            self.detections_array[self.camera_num] = []
            self.array_for_frames[self.camera_num] = []
    
    def save_frames(self):
        # This method will save the least blurry picture
        # taken from both the top and bottom cameras. 

        # Finds the least blurry image's index
        top_cam_idx = 0
        bot_cam_idx = 1
        least_blurry_indx_top = least_blurry_image_indx(self.array_for_frames[top_cam_idx])
        least_blurry_indx_bot = least_blurry_image_indx(self.array_for_frames[bot_cam_idx])

        # Save the frame with the shoes
        bottom_frame = self.array_for_frames[bot_cam_idx][least_blurry_indx_bot]
        file_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        save_directory = 'bottom_cam_images/'
        frame_filename = f'{file_timestamp}.jpg'
        save_path = save_directory + frame_filename
        cv2.imwrite(save_path, bottom_frame)

        # Save the frame with the person
        top_frame = self.array_for_frames[self.camera_num][least_blurry_indx_top]
        save_directory = 'top_cam_images/'
        save_path = save_directory + frame_filename
        cv2.imwrite(save_path, top_frame)
        return None
    
    def update_seen_ids(self):
        # Updates the seen_ids and deletes the aged tracker_ids 
        timestamp = datetime.datetime.now()
        self.seen_track_ids = [elem for elem in self.seen_track_ids if ((timestamp - elem[1]) < datetime.timedelta(minutes=2))]