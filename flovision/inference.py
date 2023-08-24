# Make a general class for inference system, which will be inherited by the entrance and laser inference system

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

from .bbox_gui import create_bounding_boxes, load_bounding_boxes
from .video import draw_border, region_dimensions, vStream, least_blurry_image_indx, get_device_indices
from .comms import sendImageToServer
from .utils import get_highest_index, findLocalServer
from .jetson import Jetson
from .face_id import face_recog

class InferenceSystem:
    """
    General class for inference system
    """
    # def __init__(self, model_name, video_res, border_thickness, display, save, bboxes, num_devices, model_type='custom', model_directory="./", model_source='local', detected_items=[], server_IP='local', annotate=False) -> None:
    def __init__(self, **kwargs) -> None:
        model_name = kwargs.get('model_name')
        video_res = kwargs.get('video_res')
        border_thickness = kwargs.get('border_thickness')
        display = kwargs.get('display')
        save = kwargs.get('save')
        bboxes = kwargs.get('bboxes')
        num_devices = kwargs.get('num_devices')
        model_type = kwargs.get('model_type', 'custom')
        model_directory = kwargs.get('model_directory', "./")
        model_source = kwargs.get('model_source', 'local')
        detected_items = kwargs.get('detected_items', [])
        server_IP = kwargs.get('server_IP', 'local')
        annotate = kwargs.get('annotate', False)
        
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
            model_source: source of the model, usually custom
            detected_items: list of items to be detected
            server_IP: specify which server to send violations to
            annotate: set true to draw detection boxes in images

        return:
            None
        """

        if server_IP == "local":
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
            zone_polygons.append(coordinates)

        # TODO: improve zone management
        # 1. Need several zones per camera
        # 2. Need to neatly store the coordinates - not just coordinates0.json and coordinates1.json

        # Load the model
        self.model = torch.hub.load(model_directory, model_type, path=model_name, force_reload=True,source=model_source, device='0') \
                    if model_type == 'custom' else torch.hub.load(model_directory, model_name, device='0', force_reload=True)
        
        # Create ByteTracker objects for each camera feed
        self.trackers = [sv.ByteTrack() for i in range(num_devices)]

        # Set frame params
        self.frame_width = video_res[0]
        self.frame_height = video_res[1]
        self.frame_size = (self.frame_width, self.frame_height)
        self.border_thickness = border_thickness
        self.display = display
        self.save = save
        self.annotate = annotate
        self.consecutive_frames_cnt = [0 for i in range(len(self.cams))]


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

        self.box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=1)

        # Directory to save the frames - for training
        self.save_dir = Path.cwd().parent / 'saved_frames'
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Directories for each item to be detected
        self.items = detected_items
        self.item_dirs = [self.save_dir / item for item in detected_items]
        [item_dir.mkdir(parents=True, exist_ok=True) for item_dir in self.item_dirs]

        # TODO: add functionality to save text boxes along with the frames

    def ByteTracker_implementation(self, detections, byteTracker):
        # byteTracker is the sv.ByteTrack() object 
        byte_tracker = byteTracker
        new_detections = []
        if len(detections) != 0:
            new_detections = byte_tracker.update_with_detections(detections)
            #Sort both detections and new_detections based on confidence scores
            sorted_original_indices = np.argsort(detections.confidence)
            sorted_new_indices = np.argsort(new_detections.confidence)
            # Create a mapping of indices between the sorted original detections and the sorted new detections
            index_mapping = dict(zip(sorted_new_indices, sorted_original_indices))
            # Update the class_ids in new_detections based on this mapping
            for new_idx, original_idx in index_mapping.items():
                new_detections.class_id[new_idx] = detections.class_id[original_idx]
        return new_detections

    def stop(self):
        """
        Stop the inference system
        """
        print("Stopping detections and releasing cameras")
        for cam in self.cams:
            cam.capture.release()

        self.jetson.cleanup()

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

        print("Starting inference system")

        zone_count = 0 # count the number of zones
        self.num_consecutive_frames = 3 # num_consecutive_frames that we want to window (to reduce jitter)

        # List of lists, beacause for each camera feed, we have a list of detections we save to choose the prevalent one as jitter reduction technique (aka goggle, no_goggle, goggle -> goggle )
        self.detections_array = [[] for _ in range(len(self.zones))]

        # List of lists, beacause for each camera feed, we have a list of frames we save to choose the least blurry one
        self.array_for_frames = [[] for _ in range(len(self.zones))]

        # Need a trigger flag for each of the camera feeds. Initialized to False
        self.detection_trigger_flag = [False for _ in range(len(self.zones))]

        # Split detections into different sets depending on object, by bounding box (aka [goggles, no_goggles])
        self.present_indices = [2* idx for idx in range(len(self.items))]
        self.absent_indices = [2* idx + 1 for idx in range(len(self.items))]
        item_sets = [[present, absent] for present, absent in zip(self.present_indices, self.absent_indices)]

        while True:
            try:
                ### Saving a frame from each camera feed ###
                self.captures = []
                frame_unavailable = False
                for cam in self.cams:
                    ret, frame = cam.getFrame()
                    if not ret:
                        frame_unavailable = True
                        break

                    self.captures.append(frame)

                if frame_unavailable:
                    continue
              

                ##### Iterating over frames saved from each of the connected cameras #####
                
                self.camera_num = 0 # the index of the vide stream being processed
                # Iterate over cameras, 1 frame from each  
                for frame in self.captures:
                    # Send through the model
                    results = self.model(frame)
                    # Convert the detections to the Supervision-compatible format
                    self.detections = sv.Detections.from_yolov5(results)
                    # Run NMS to remove double detections
                    self.detections = self.detections.with_nms(threshold=iou_thres,  class_agnostic=agnostic_nms)  # apply NMS to detections
                    # If detections present, track them. Assign track_id
                    if len(self.detections) > 0:
                        self.detections = self.ByteTracker_implementation(detections=self.detections, byteTracker=self.trackers[self.camera_num])

                    if self.use_zones:
                        # Check # of detections in a zone (We are assuming there's 1 zone per camera - TODO: UPGRADE TO MULTIPLE)
                        mask = self.zones[self.camera_num].trigger(detections=self.detections) #this changes self.zones.current_count
                    
                    # Annotate the zones and the detections on the frame if the flag is set
                    if self.annotate:
                        frame = self.box_annotator.annotate(scene=frame, detections=self.detections)
                        frame = self.zone_annotators[self.camera_num].annotate(scene=frame)

                    # FIX THIS LOGIC TO ALSO WORK IF MASK IS NOT PRESENT
                    # Split into different sets of detections depending on object, by bounding box (aka tuple(goggles/no_goggles, shoes/no_shoes) )
                    self.item_detections = tuple([self.detections[mask & np.isin(self.detections.class_id, item_sets[i])] for i in range(len(item_sets))])

                    # TRIGGER EVENT - TODO: ADD MULTIPLE TRIGGER EVENTS FUNCTIONALITY
                    if self.trigger_event():
                        self.detection_trigger_flag[self.camera_num] = True
                        
                        results_dict = results.pandas().xyxy[0].to_dict()
                        results_json = json.dumps(results_dict)
                        print()
                        print("RESULTS: ", results)
                        print("RESULTS JSON: ", results_json)
                        print()

                    if self.detection_trigger_flag[self.camera_num]:
                        self.trigger_action()

                    self.other_actions()
                
                    # Display frame
                    if self.display:
                        cv2.imshow(f'Camera {self.cams[self.camera_num].src}', frame)

                    # Update iteration index for the loop    
                    self.camera_num += 1

            except Exception as e:
                print("Frame unavailable, error was: ", e)
                traceback.print_exc()

            if cv2.waitKey(1) == ord('q'):
                self.stop()

    def trigger_event(self) -> bool:
        """
        Determine if a trigger event has occurred

        return:
            True if a trigger event has occurred, False otherwise
        """
        raise NotImplementedError("Trigger event not implemented")
    
    def trigger_action(self) -> None:
        """
        Perform the trigger action
        """
        raise NotImplementedError("Trigger action not implemented")
    

    def other_actions(self) -> None:
        """
        Perform other actions
        """
        pass



class Violation():
    # A class that holds everything needed to hold information
    # for a violation on a certain track_id. Violation objects
    # expected to be stored in a dictionary as a value with 
    # key being the track_id number. 
    def __init__(self, camera_id, class_id, timestamp, violation_code) -> None:
        self.camera_id = camera_id
        self.class_id = []
        self.class_id.append(class_id)
        self.timestamps = []
        self.timestamps.append(timestamp)
        self.violation_codes = []
        self.violation_codes.append(violation_code)
    
    def Check_Code(self, violation_code, class_id) -> bool:
        # Will check if the there's already an existing violation
        # with matching class ID that caused it. 
        self.Update_Time()
        return violation_code in self.violation_codes and class_id in self.class_id
    
    def Add_Code(self, violation_code, class_id) -> None:
        # Will add a new violation code to the violation object
        self.violation_codes.append(violation_code)
        self.class_id.append(class_id)
        self.timestamps.append(datetime.datetime.now())

    def Update_Time(self) -> bool:
        # Will update the timestamps recorded here and will delete
        # the timestamps that are over 10 minutes old. This method
        # should be the only method used to delete the violation
        # codes stored in a violation object. 
        if len(self.timestamps) == 0:
            return False

        # 3 list comprehensions used to update the timestamps and 
        # violation codes. The condition list comprehension will
        # store a list of booleans of which elements should stay
        # or not. Using that list, the timestamps and 
        # violation_codes list will be told to keep the elements
        # that correlate to the True elements of the condition
        # list. 
        condition = [((datetime.datetime.now() - timestamp) < datetime.timedelta(minutes=10)) for timestamp in self.timestamps]
        self.timestamps = [timestamp for timestamp, cond in zip(self.timestamps, condition) if cond]
        self.violation_codes = [violation_code for violation_code, cond in zip(self.violation_codes, condition) if cond]
        
        if len(self.timestamps) == 0:
            return False
        return True

    def __len__(self) -> int:
        # Will return the number of valid timestamps available
        return self.Update_Time()