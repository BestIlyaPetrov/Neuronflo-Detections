
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
from ..inference import InferenceSystem

class EntranceInferenceSystem(InferenceSystem):
    def __init__(self, *args, **kwargs) -> None:


        # If need to overwrite a particular argument do the following. 
        # Let's say need to overwrite the 'model_directory' argument
        # kwargs['model_directory'] = 'my_new_directory_path'

        super().__init__(*args, **kwargs)
        self.zone_count = [0 for i in range(len(self.cams))]
        self.consecutive_frames_cnt = [0 for i in range(len(self.cams))]


    def trigger_event(self) -> bool:

        return int(self.zones[self.camera_num].current_count) > self.zone_count[self.camera_num] and (not self.detection_trigger_flag[self.camera_num]) 
    
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

            # For each detected item (aka goggles vs no_goggles being 1 item), append detected class_id from the last N frames 
            the_detections = [[] for _ in range(len(self.items))]
            for detected_items in self.detections_array[self.camera_num]:
                for the_item in detected_items:
                    if hasattr(the_item, 'class_id') and len(the_item.class_id) > 0:
                        [the_detections[detected_items.index(the_item)].append(int(ids)) for ids in the_item.class_id]
            # Check the most common detection class_id and choose prevalent one (aka goggle, no_goggle, goggle, goggle -> goggle)
            most_common_detections = self.present_indices
            for i in range(len(self.items)):
                if len(the_detections[i]):
                    most_common_detections[i] = collections.Counter(the_detections[i]).most_common(1)[0][0]
                    print(f"Most common detection for object {self.items[i]} is {most_common_detections[i]}")
                else:
                    print(f"No detections for object {self.items[i]}")
                
            # Pick the least blurry image to send to the server (we assume images don't vary that much within a small enough del_t or in this case N = self.num_consecutive_frames)
            least_blurry_indx = least_blurry_image_indx(self.array_for_frames[self.camera_num])

            # Compliance Logic
            compliant = False
            if most_common_detections[0] == 0:
                compliant =  False

            elif most_common_detections[0] == 1:
                compliant = True

            bordered_frame = draw_border(self.array_for_frames[self.camera_num][least_blurry_indx],  compliant, self.border_thickness)

            # Pack the data for the server to process - TODO: figure out what data we are sending to the server - what can we gather?
            data = {
                'zone_name': '1',
                'crossing_type': 'coming', #or leaving
                'compliant': str(compliant)
            }

            # send the actual image to the server
            sendImageToServer(bordered_frame, data, IP_address=self.server_IP)

            # Save the image locally for further model retraining
            if self.save:
                self.save_frames(self.array_for_frames[self.camera_num])

            # Reset the arrays for the data and the images, since we just sent it to the server
            self.detections_array[self.camera_num] = []
            self.array_for_frames[self.camera_num] = []
    
    def other_actions(self) -> None:
        for i in range(len(self.zones)):
            self.zone_count[i] = self.zones[i].current_count