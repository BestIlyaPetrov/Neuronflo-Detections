# Make a general class for inference system, which will be inherited by the entrance and laser inference system

from threading import Thread
import cv2
import time
import numpy as np
import torch, torchvision
import json
import datetime
import time
import os

import supervision as sv
# print(f"Supervision contents are {dir(sv)}")
import traceback
import collections
from pathlib import Path

from .bbox_gui import create_bounding_boxes, load_bounding_boxes
from .video import draw_border, region_dimensions, vStream, least_blurry_image_indx, get_device_indices, adjust_color
from .comms import sendImageToServer
from .utils import get_highest_index, findLocalServer
from .jetson import Jetson
from .face_id import face_recog
from .record import Recorder

# Initialize some constants
BYTETRACKER_MATCH_THRESH = 0.4
CONFIDENCE_THRESH = 0.4

# Run method params
NUM_CONSECUTIVE_FRAMES = 3
TRACK_ID_KEEP_ALIVE = 1 # minutes

LINE_START = sv.Point(320, 0)
LINE_END = sv.Point(320, 480)


class InferenceSystem:
    """
    General class for inference system
    """

    def __init__(self, **kwargs) -> None:
        model_name = kwargs.get('model_name')
        video_res = kwargs.get('video_res')
        display = kwargs.get('display')
        save = kwargs.get('save')
        bboxes = kwargs.get('bboxes')
        num_devices = kwargs.get('num_devices')
        model_type = kwargs.get('model_type', 'custom')
        model_directory = kwargs.get('model_directory', "./")
        model_source = kwargs.get('model_source', 'local')
        detected_items = kwargs.get('detected_items', [])
        server_IP = kwargs.get('server_IP', 'local')
        annotate_raw = kwargs.get('annotate_raw', False)
        annotate_violation = kwargs.get('annotate_violation', False)
        debug = kwargs.get('debug', False)
        record = kwargs.get('record', False)
        self.adjust_brightness = kwargs.get('adjust_brightness', False)
        self.save_labels = kwargs.get('save_labels', False)
        self.use_nms = kwargs.get('use_nms', True)

        print("\n\n##################################")
        print("PARAMETERS INSIDE INFERENCE.PY\n")
        print(f"model_name: {model_name}")
        print(f"video_res: {video_res}")
        print(f"display: {display}")
        print(f"save: {save}")
        print(f"bboxes: {bboxes}")
        print(f"num_devices: {num_devices}")
        print(f"model_type: {model_type}")
        print(f"model_directory: {model_directory}")
        print(f"model_source: {model_source}")
        print(f"detected_items: {detected_items}")
        print(f"server_IP: {server_IP}")
        print(f"annotate_raw: {annotate_raw}")
        print(f"annotate_violation: {annotate_violation}")
        print(f"debug: {debug}")
        print(f"record: {record}")
        print(f"adjust_brightness: {self.adjust_brightness}")
        print(f"save_labels: {self.save_labels}")
        print(f"use_nms: {self.use_nms}")
        print("##################################\n\n")

        """
        param:
            model_name: name of the model to be used for inference
            video_res: resolution of the video
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
        # Verbose mode True/False
        self.debug = debug

        if server_IP == "local":
            self.server_IP = findLocalServer()
        else:
            self.server_IP = server_IP

        cap_index = get_device_indices(quantity = num_devices)

        # Initialize the cameras
        self.cams = [vStream(cap_index[i], i, video_res) for i in range(num_devices)]

        # Initialize the jetson's peripherals and GPIO pins
        self.jetson = Jetson()

        # Define the detection regions
        zone_polygons = []

        func = create_bounding_boxes if bboxes else load_bounding_boxes
        for i, cam in enumerate(self.cams):
            coordinates_set = func(cam)
            for j, coordinates in enumerate(coordinates_set):
                zone_polygons.append(Zone(i, j, coordinates, tuple(video_res)))
        
        # Initialize the zone polygons - list of custom zone objects defined in Zone() at the bottom
        self.zone_polygons = zone_polygons

        # Load the model
        self.model = torch.hub.load(model_directory, model_type, path=model_name, force_reload=True,source=model_source, device='0') \
                    if model_type == 'custom' else torch.hub.load(model_directory, model_name, device='0', force_reload=True)
        
        self.model.eval() #set the model into eval mode

        # Create ByteTracker objects for each camera feed
        self.trackers = [sv.ByteTrack(match_thresh=BYTETRACKER_MATCH_THRESH) for i in range(num_devices)]

        # Set frame params
        self.frame_width = video_res[0]
        self.frame_height = video_res[1]
        self.frame_size = (self.frame_width, self.frame_height)
        self.display = display
        self.save = save
        self.annotate_raw = annotate_raw
        self.annotate_violation = annotate_violation
        self.consecutive_frames_cnt = [0 for i in range(len(self.cams))] # counter for #of frames taken after trigger event
        self.num_consecutive_frames = NUM_CONSECUTIVE_FRAMES # num_consecutive_frames that we want to window (to reduce jitter)


        # Set the zone params
        colors = sv.ColorPalette.default()
        # self.zones = [
        #     sv.PolygonZone(
        #         polygon=zone_object.polygon,
        #         frame_resolution_wh=self.frame_size,
        #         triggering_position=sv.Position.CENTER
        #     )
        #     for zone_object
        #     in zone_polygons
        # ]
        self.zone_annotators = [
            sv.PolygonZoneAnnotator(
                zone=zone.PolyZone,
                color=colors.by_idx(index + 2),
                thickness=1,
                text_thickness=1,
                text_scale=1
            )
            for index, zone
            in enumerate(self.zone_polygons)
        ]
        # Intialize the annotators
        self.box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=1)
        self.label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)
        self.blur_annotator = sv.BlurAnnotator()
        line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
        line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

        

        # Directory to save the frames - for training
        # self.save_dir = Path.cwd().parent / 'saved_frames'
        # self.save_dir.mkdir(parents=True, exist_ok=True)

        # Directories for each item to be detected
        self.items = detected_items

        # I don't think we need this functionality anymore - ilya
        # self.item_dirs = [self.save_dir / item for item in detected_items]
        # [item_dir.mkdir(parents=True, exist_ok=True) for item_dir in self.item_dirs]

        # Decides if the last 5 seconds should be stored
        self.record = record
        if self.record:
            self.recorder = Recorder(system=self)
        


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

    def save_frames(self,frame, detections, cam_idx):
        """
        Save the frames to the disk
        """
        try:
            cam_dir = Path(os.path.abspath(__file__)).parent / 'saved_frames' / f'cam{cam_idx}'
            cam_dir.mkdir(parents=True, exist_ok=True)
            item_count = get_highest_index(str(cam_dir)) + 1
            cv2.imwrite(str(cam_dir / f'img_{item_count:04d}.jpg'), frame)

            if self.save_labels:
                label_dir = Path(os.path.abspath(__file__)).parent / 'saved_frames' / f'cam{cam_idx}_labels'
                label_dir.mkdir(parents=True, exist_ok=True)

                # write the labels in yolo format to the text file
                with open(str(label_dir / f'img_{item_count:04d}.txt'), 'w') as file:
                    for i in range(len(detections['xyxy'])):
                        # Convert from xyxy to yolo format
                        x1, y1, x2, y2 = detections['xyxy'][i]
                        x_center = (x1 + x2) / 2 / self.frame_width
                        y_center = (y1 + y2) / 2 / self.frame_height
                        width = (x2 - x1) / self.frame_width
                        height = (y2 - y1) / self.frame_height
                        
                        class_id = detections['class_id'][i]
                        
                        file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        except Exception as e:
            print("Error saving frames. Error was:\n{e}")
            traceback.print_exc()

    def run(self, **kwargs) -> None:
        iou_thres = kwargs.get('iou_thres', 0.7)
        agnostic_nms = kwargs.get('agnostic_nms', False)

        """
        param:
            iou_thres: iou threshold
            agnostic_nms: agnostic nms

        return:
            None
        """

        print("Starting inference system")

        """
        # List of lists, beacause for each camera, we have a list of detections we save to choose the prevalent one as jitter reduction technique (aka goggle, no_goggle, goggle -> goggle )
        self.detections_array = [[] for _ in range(len(self.cams))]

        # List of lists, beacause for each camera, we have a list of frames we save to choose the least blurry one
        self.array_for_frames = [[] for _ in range(len(self.cams))]
        """
        # Need a trigger flag for each of the zones. Initialized to False
        self.detection_trigger_flag = [False for _ in range(len(self.cams))]

        # Split detections into different sets depending on object, by bounding box (aka [goggles, no_goggles])
        
        # MAYBE USE THIS IN CONJUNCTION WITH SIALOI's CODE
        # self.present_indices = [2* idx for idx in range(len(self.items))]
        # self.absent_indices = [2* idx + 1 for idx in range(len(self.items))]
        # item_sets = [[present, absent] for present, absent in zip(self.present_indices, self.absent_indices)]
        
        class_names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        print(f"Class names: {class_names}")

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
                if len(self.captures) < len(self.cams):
                    frame_unavailable = True
            

                if frame_unavailable:
                    # if self.debug:
                    #     print("\n\n##########################################################################")
                    #     print("###########FRAME WAS UNAVAILABLE SO SKIPPING TO NEXT ITERATION ############")
                    #     print("##########################################################################\n\n")

                    continue

                ##### Iterating over frames saved from each of the connected cameras #####
                # If record flag raised, store frames 
                if self.record:
                    self.recorder.store()

                self.camera_num = 0 # the index of the video stream being processed
                # Iterate over cameras, 1 frame from each  
                for frame in self.captures:
                    #Print which camera we are processing
                    # Send through the model
                    detection_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if self.adjust_brightness:
                        detection_frame = adjust_color(detection_frame)

                    results = self.model(detection_frame)

                    # Convert the results to a numpy array in the format [x_min, y_min, x_max, y_max, confidence, class]
                    predictions = results.xyxy[0].cpu().numpy()

                    # Define the confidence threshold
                    conf_thresh = CONFIDENCE_THRESH #0.4

                    # Filter out predictions with a confidence score below the threshold
                    filtered_predictions = predictions[predictions[:, 4] > conf_thresh]

                    # Load the filtered predictions back into the results object
                    results.pred[0] = torch.tensor(filtered_predictions)
                    
                    # Convert the detections to the Supervision-compatible format
                    self.detections = sv.Detections.from_yolov5(results)
                    
                    if self.use_nms:
                        # Run NMS to remove double detections
                        self.detections = self.detections.with_nms(threshold=iou_thres,  class_agnostic=agnostic_nms)

                    # If detections present, track them. Assign track_id
                    if len(self.detections) > 0:
                        self.detections = self.ByteTracker_implementation(detections=self.detections, byteTracker=self.trackers[self.camera_num])
        
                    """
                    # Check which zone each detection belongs to
                    zone_detections = []
                    for detection in self.detections:
                        curr_detections = []
                        # Calculate the center of the bounding box
                        center_x = (detection.x_min + detection.x_max) / 2
                        center_y = (detection.y_min + detection.y_max) / 2

                        # Check which zone the center of the bounding box falls into
                        for idx, zone in enumerate(self.zones):
                            if zone.contains(center_x, center_y):
                                curr_detections.append(detection)

                        zone_detections.append(curr_detections)
                    
                    masks = [zone.PolyZone.trigger(detections=zone_detections[j]) for j, zone in enumerate(self.zone_polygons) if len(zone_detections[j]) > 0]

                    # Take intersection of all the masks
                    mask = np.all(masks, axis=0)
                    """
                    self.masks  = [zone.PolyZone.trigger(detections=self.detections) for zone in self.zone_polygons]
                    # Annotate the zones and the detections on the frame if the flag is set
                    if self.annotate_raw:
                        annotated_frame = frame.copy()
                        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=self.detections)
                        # annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=self.detections)
                        for zone_annotator, zone in zip(self.zone_annotators, self.zone_polygons):
                            annotated_frame = zone_annotator.annotate(scene=annotated_frame) if zone.camera_id == self.camera_num else annotated_frame
                        # annotated_frame = self.zone_annotators[self.camera_num].annotate(scene=annotated_frame)

                    # MAYBE USE THIS IN CONJUNCTION WITH SIALOI's CODE
                    # Split into different sets of detections depending on object, by bounding box (aka tuple(goggles/no_goggles, shoes/no_shoes) )
                    # self.item_detections = tuple([self.detections[mask & np.isin(self.detections.class_id, item_sets[i])] for i in range(len(item_sets))])

                    # TRIGGER EVENT - TODO: ADD MULTIPLE TRIGGER EVENTS FUNCTIONALITY
                    if self.trigger_event():
                        self.detection_trigger_flag[self.camera_num] = True
                        
                        if self.debug:
                            results_dict = results.pandas().xyxy[0].to_dict()
                            # results_json = json.dumps(results_dict) #don't delete this line pls
                            print()
                            print("TRIGGER EVENT - THESE WERE THE DETECTIONS:")
                            print(results)
                            print(results_dict)
                            # print("RESULTS JSON: ", results_json)
                            print()

                    if self.detection_trigger_flag[self.camera_num]:
                        self.trigger_action()
                        # After the image is sent to the server
                        if not self.detection_trigger_flag[self.camera_num]:
                            # Display annotated frame with violations highlighted 
                            if self.annotate_violation:
                                # violation_frame = self.annotate_violations()
                                # cv2.imshow("Violation Sent", self.frame_with_violation)
                                pass
                            if self.record:
                                # This will be the command for sending in the before and after footage of the violation
                                self.recorder.send()

                            # # Reset the arrays for the data and the images, since we just sent it to the server
                            # self.array_for_frames = [[[] for _ in range(len(self.num_consecutive_frames))] for _ in range(len(self.cams))]
                            # self.detections_array = [[[] for _ in range(len(self.num_consecutive_frames))] for _ in range(len(self.cams))]
                            # self.violations_array = [[[] for _ in range(len(self.num_consecutive_frames))] for _ in range(len(self.cams))]

                            if self.debug:
                                print("CLEARED MAIN ARRAYS")


                    self.other_actions()
                
                    # Display frame
                    if self.display:
                        cv2.imshow(f'Camera {self.cams[self.camera_num].src}', annotated_frame)

                    # Update iteration index for the loop    
                    self.camera_num += 1

            except Exception as e:
                print("Frame unavailable, error was: ", e)
                traceback.print_exc()
                self.initialize_system()
            
            # If frame is being displayed
            if self.display:
                if cv2.waitKey(1) == ord('q'):
                    self.stop()

    def initialize_system(self) -> None:
        """
        Initialize the system with this method.
        inference.py will call this method when an exception occurs to reset all arrays
        """
        raise NotImplementedError("Initialize function not implemented")
    
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
    
    def annotate_violations(self) -> list:
        """
        Implement the desired annotation method
        """
        raise NotImplementedError("Annotate violations not implemented")

    def other_actions(self) -> None:
        """
        Perform other actions
        """
        pass


class Zone():
    def __init__(self, camera_id, zone_id, polygon, frame_size) -> None:
        self.camera_id = camera_id
        self.zone_id = zone_id
        self.polygon = polygon
        self.PolyZone = sv.PolygonZone(
                polygon=polygon,
                frame_resolution_wh=frame_size,
                triggering_position=sv.Position.CENTER)
        self.last_count = 0

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
        self.zone = None # Will be set to the zone object that the violation occured in.
    
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
        condition = [((datetime.datetime.now() - timestamp) < datetime.timedelta(minutes=TRACK_ID_KEEP_ALIVE)) for timestamp in self.timestamps]
        self.timestamps = [timestamp for timestamp, cond in zip(self.timestamps, condition) if cond]
        self.violation_codes = [violation_code for violation_code, cond in zip(self.violation_codes, condition) if cond]
        
        if len(self.timestamps) == 0:
            return False
        return True

    def __len__(self) -> int:
        # Will return the number of valid timestamps available
        self.Update_Time()
        return len(self.timestamps)
    
    def Get_Timestamp(self, violation_code) -> datetime:
        index = self.violation_codes.index(violation_code)
        return self.timestamps[index]