# Make a general class for inference system, which will be inherited by the entrance and laser inference system

from threading import Thread
import cv2
import time
import numpy as np
import torch, torchvision
import json

import supervision as sv
import traceback
import collections
from pathlib import Path

from .bbox_gui import create_bounding_boxes, load_bounding_boxes
from .video import draw_border, region_dimensions, vStream, least_blurry_image_indx, get_device_indices
from .comms import sendImageToServer
from .utils import get_highest_index, findLocalServer
from .jetson import Jetson
from zeroconf import ServiceBrowser, Zeroconf

class InferenceSystem:
    """
    General class for inference system
    """
    def __init__(self, model_name, video_res, border_thickness, display, save, bboxes, num_devices, model_type, model_directory="./", model_source='local', detected_items=[]) -> None:
        """
        param:
            model_name: name of the model to be used for inference
            video_res: resolution of the video
            border_thickness: thickness of the border around the region of interest
            display: whether to display the video or not
            save: whether to save the video or not
            bboxes: whether to use bounding boxes or not
            num_devices: number of devices to be used for inference
            model_type: type of the model to be used for inference
            model_directory: directory where the model is stored
            model_source: source of the model
            detected_items: list of items to be detected

        return:
            None
        """

        self.server_IP = findLocalServer()
        cap_index = get_device_indices(quantity = num_devices)

        # Initialize the cameras
        self.cams = [vStream(cap_index[i], video_res) for i in range(num_devices)]

        # Initialize the jetson's peripherals and GPIO pins
        self.jetson = Jetson()

        # Define the detection regions
        zone_polygons = []

        func = create_bounding_boxes if bboxes else load_bounding_boxes
        for i, cam in enumerate(self.cams):
            coordinates = func(cam)
            coordinates[:,0] += i*video_res[0] # shift the x coordinates
            zone_polygons.append(coordinates)

        # Load the model
        self.model = torch.hub.load(model_directory, model_source, path=model_name, force_reload=True,source=model_source, device='0') \
                    if model_type == 'custom' else torch.hub.load(model_directory, model_name, device='0')
        
        # Set frame params
        self.frame_width = video_res[0] * num_devices
        self.frame_height = video_res[1]
        self.frame_size = (self.frame_width, self.frame_height)
        self.border_thickness = border_thickness
        self.display = display
        self.save = save

        # Set the zone params
        colors = sv.ColorPalette.default()
        self.zones = [
            sv.PolygonZone(
                polygon=polygon,
                frame_resolution_wh=self.frame_size
            )
            for polygon
            in zone_polygons
        ]
        self.zone_annotators = [
            sv.PolygonZoneAnnotator(
                zone=zone,
                color=colors.by_idx(index + 2),
                thickness=4,
                text_thickness=2,
                text_scale=2
            )
            for index, zone
            in enumerate(self.zones)
        ]

        self.box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)

        # Directory to save the frames
        self.save_dir = Path.cwd().parent / 'saved_frames'
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Directories for each item to be detected
        self.items = detected_items
        self.item_dirs = [self.save_dir / item for item in detected_items]
        [item_dir.mkdir(parents=True, exist_ok=True) for item_dir in self.item_dirs]

    def stop(self):
        """
        Stop the inference system
        """
        print("Stopping detections and releasing cameras")
        for cam in self.cams:
            cam.capture.release()

        cv2.destroyAllWindows()
        exit(1)

    def save_frames(self,frame_arr):
        """
        Save the frames to the disk
        """

        try:
            for i, item in enumerate(frame_arr):
                item_count = get_highest_index(self.item_dirs[i]) + 1
                for img in item:
                    cv2.imwrite(str(self.item_dirs[i] / f'{self.items[i]}_img_{item_count:04d}.jpg'), img)
                    item_count += 1


        except Exception as e:
            print("Error saving frames")
            print(e)
            traceback.print_exc()

    def run(self, iou_thres, agnostic_nms):
        """
        param:
            iou_thres: iou threshold
            agnostic_nms: agnostic nms

        return:
            None
        """

        raise NotImplementedError


