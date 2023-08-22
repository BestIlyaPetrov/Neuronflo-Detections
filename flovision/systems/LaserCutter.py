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

class LaserInferenceSystem(InferenceSystem):
    def __init__(self, model_name, video_res, border_thickness, display, save, bboxes, num_devices, model_type, model_directory="./", model_source='local', detected_items=[]) -> None:
        
        super().__init__(model_name, video_res, border_thickness, display, save, bboxes, num_devices, model_type, model_directory, model_source, detected_items)

    def trigger_event(self) -> bool:
        # Trigger event for laser
        return self.zones[0].current_count >= 1
    
    def trigger_action(self) -> None:
        print("Laser Cutter inference successfully launched")
        self.detection_trigger_flag = False
        byte_tracker = sv.ByteTracker()
      
        # Run Inference
        results = self.model(self.frame)

        # load results into supervision
        detections = sv.Detections.from_yolov5(results)
        
        # Apply NMS to remove double detections
        detections = detections.with_nms(threshold=iou_thres,  class_agnostic=agnostic_nms)
        
        # Check if the distance metric has been violated
        self.detection_trigger_flag, detection_info = self.trigger_event(detections=detections)
        # The detection_info variable will hold a tuple that
        # will be the indices for the objects in violation
        
        # Gives all detections ids and will be processed in the next step
        detections = self.ByteTracker_implementation(detections=detections, byteTracker=byte_tracker)
        if self.save:
            self.save_frames(self.captures)