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



class EnvisionInferenceSystem:
    def __init__(self, model_name, video_res, border_thickness, display, save, bboxes):
        self.initialize(model_name, video_res, border_thickness, display, save, bboxes)

    def initialize(self, model_name, video_res, border_thickness, display, save, bboxes):


        self.server_IP = findLocalServer()
        cap_index = get_device_indices(quantity = 1)

        # Initialize the cameras
        self.cams = []
        self.cams.append(vStream(cap_index[0], video_res))
        # self.cams.append(vStream(cap_index[1], video_res))

        # Initialize Jetson peripherals
        self.jetson = Jetson()

        # Load the model
        self.model = torch.hub.load('./', 'custom', path=model_name, force_reload=True, source='local', device='0')
        print(f"The names of the different classes are: {self.model.names}")
        # self.model.classes = [0]  # Set the desired class


        # Set frame params
        self.frame_size = (video_res[0], video_res[1])  
        self.border_thickness = border_thickness
        self.display = display
        self.save = save

        self.box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)

    def stop(self):
        print("Stopping detections and releasing cameras")
        for i in range(0, len(self.cams)):
            self.cams[i].capture.release()
        cv2.destroyAllWindows()
        exit(1)


    def save_frames(self, frame_arr):
        try:
            # Path of the directory
            dir_path = Path.cwd().parent / 'saved_frames'
            dir_path.mkdir(parents=True, exist_ok=True)
            # Create the directories if it doesn't exist
            count = get_highest_index(dir_path) + 1 
            for img in frame_arr[0]:
                cv2.imwrite(str(dir_path / f"laser_img_{count:04d}.jpg"), img)
                print(f"Saved image: laser_img_{count:04d}.jpg")
                count += 1 
        except Exception as e:
            print("Couldn't save the frames.")
            print(e)
            traceback.print_exc()

    def run(self, iou_thres, agnostic_nms):
        print("Inference successfully launched")
        zone_count = 0
        n = 3 # num_consecutive_frames that we want to window (to reduce jitter)
        detections_array = []
        frame1_array = []
        frame2_array = []
        cnt = 0
        n_pics = 20
        detection_trigger_flag = False
        trigger_start = 0
        delta_T = 5 #seconds
        # fps_list = []
        while True:
            try:
                # start the timer
                # start_time = time.time()

                ## Make the slowest cam be the bottleneck here
                new_frame1, myFrame1 = self.cams[0].getFrame()
                if new_frame1 == False:
                    continue
                # new_frame2, myFrame2 = self.cams[1].getFrame()
                # if new_frame2 == False:
                #     continue

                # frame = np.hstack((myFrame1, myFrame2))
                frame = myFrame1.copy()

                # Run Inference
                results = self.model(frame)

                # load results into supervision
                detections = sv.Detections.from_yolov5(results)
                # Apply NMS to remove double detections
                detections = detections.with_nms(threshold=iou_thres,  class_agnostic=agnostic_nms)  # apply NMS to detections
                labels = detections.labels
                cnt = 0
                violation_detected = False
                for label in labels:
                    if label == 0: # Assuming class 0 is "soldering"
                        minX, minY, maxX, maxY = detections.boxes[cnt]
                        centerX, centerY = findCenter(minX=minX, minY=minY, maxX=maxX, maxY=maxY)
                        cnt2 = 0
                        for label2 in labels:
                            if label2 == 1: # Assuming class 1 is "no_goggles"
                                minX2, minY2, maxX2, maxY2 = detections.boxes[cnt2]
                                centerX2, centerY2 = findCenter(minX=minX2, minY=minY2, maxX=maxX2, maxY=maxY2)
                                distX = abs(centerX - centerX2)/self.frame_size[0]
                                distY = abs(centerY - centerY2)/self.frame_size[1]
                                if distX < 0.2 and distY < 0.2:
                                    violation_detected = True
                                    break
                            cnt2 = cnt2 + 1
                    if violation_detected:
                        break
                    cnt = cnt + 1

                

                # Annotate
                # mask = []
                # for zone, zone_annotator in zip(self.zones, self.zone_annotators):
                #     mask.append(zone.trigger(detections=detections))
                #     frame = self.box_annotator.annotate(scene=frame, detections=detections)
                #     frame = zone_annotator.annotate(scene=frame)

                # Split into two sets of detections by bounding box
                # goggle_classes_set = [0, 1] #goggles, no_goggles
                # shoe_classes_set = [2, 3] #safe_shoes, not_safe_shoes
                # goggle_det = detections[mask[0] & np.isin(detections.class_id, goggle_classes_set)] # mask[0] is the left bounding box
                # shoe_det = detections[mask[1] & np.isin(detections.class_id, shoe_classes_set)] # mask[1] is the right bounding box

                
                if self.zones[0].current_count >= 1:
                    frame1_array.append(myFrame1)
                    if self.save:
                        self.save_frames([frame1_array])

                # Display frame
                if self.display:
                    cv2.imshow('ComboCam', frame)

            except Exception as e:
                print('frame unavailable', e)
                traceback.print_exc()

            if cv2.waitKey(1) == ord('q'):
                self.stop()
               
def findCenter(minX, minY, maxX, maxY):
    centerX = minX + (maxX - minX)/2
    centerY = minY + (maxY - minY)/2
    return centerX, centerY