# Make a general class for inference system, which will be inherited by the entrance and laser inference system

from threading import Thread
import cv2
import time
import numpy as np
import torch, torchvision
from utils.general import non_max_suppression
import json
import datetime
import time
import os

import supervision as sv
from supervision import Detections
from supervision.tracker.byte_tracker.basetrack import BaseTrack
# print(f"Supervision contents are {dir(sv)}")
import traceback
import collections
from pathlib import Path

from .bbox_gui import create_bounding_boxes, load_bounding_boxes
from .video import draw_border, region_dimensions, vStream, least_blurry_image_indx, get_camera_src, adjust_color
from .comms import sendImageToServer
from .utils import get_highest_index, findLocalServer
from .jetson import Jetson
from .face_id import face_recog
from .record import Recorder
from utils.torch_utils import select_device, smart_inference_mode




# Initialize some constants
BYTETRACKER_MATCH_THRESH = 0.95
CONFIDENCE_THRESH = 0.2

# Run method params
NUM_CONSECUTIVE_FRAMES = 3
TRACK_ID_KEEP_ALIVE = 1 # minutes
DROP_BYTETRACK_AFTER = 5 #seconds
BYTETRACKER_FULL_RESET_TIMEOUT = 5 #minutes

# #Unused at the moment
# LINE_START = sv.Point(320, 0)
# LINE_END = sv.Point(320, 480)


class InferenceSystem:
    """
    General class for inference system
    """

    def __init__(self, **kwargs) -> None:
        model_name = kwargs.get('model_name')
        video_res = kwargs.get('video_res')
        bboxes = kwargs.get('bboxes')
        num_devices = kwargs.get('num_devices')
        model_type = kwargs.get('model_type', 'custom')
        model_directory = kwargs.get('model_directory', "./")
        model_source = kwargs.get('model_source', 'local')
        cam_rotation_type = kwargs.get('cam_rotation_type', None)
        # detected_items = kwargs.get('detected_items', [])
        server_IP = kwargs.get('server_IP', 'local')
        self.display = kwargs.get('display')
        self.save = kwargs.get('save')
        self.save_labels = kwargs.get('save_labels', False)
        self.annotate_raw = kwargs.get('annotate_raw', False)
        self.annotate_violation = kwargs.get('annotate_violation', False)
        self.debug = kwargs.get('debug', False)
        self.record = kwargs.get('record', False)
        self.adjust_brightness = kwargs.get('adjust_brightness', False)
        self.use_nms = kwargs.get('use_nms', True)
        self.data_gather_only = kwargs.get('data_gather_only', False)
        self.show_fps = kwargs.get('show_fps', False)
        self.send_to_portal = kwargs.get('send_to_portal', False)

        print("\n\n##################################")
        print("PARAMETERS INSIDE INFERENCE.PY\n")
        print(f"model_name: {model_name}")
        print(f"video_res: {video_res}")
        print(f"bboxes: {bboxes}")
        print(f"num_devices: {num_devices}")
        print(f"model_type: {model_type}")
        print(f"model_directory: {model_directory}")
        print(f"model_source: {model_source}")
        # print(f"detected_items: {detected_items}")
        print(f"server_IP: {server_IP}")
        print(f"display: {self.display}")
        print(f"save: {self.save}")
        print(f"save_labels: {self.save_labels}")
        print(f"annotate_raw: {self.annotate_raw}")
        print(f"annotate_violation: {self.annotate_violation}")
        print(f"debug: {self.debug}")
        print(f"record: {self.record}")
        print(f"adjust_brightness: {self.adjust_brightness}")
        print(f"save_labels: {self.save_labels}")
        print(f"use_nms: {self.use_nms}")
        print(f"data_gather_only: {self.data_gather_only}")
        print(f"show_fps: {self.show_fps}")
        print(f"send_to_portal: {self.send_to_portal}")
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

        if server_IP == "local":
            self.server_IP = findLocalServer()
        else:
            self.server_IP = server_IP

        cap_src = get_camera_src(quantity = num_devices)

        
        # Initialize the cameras
        crop_coordinates = []
        #cam0
        crop_coordinates.append((404.0, 1044.0, 77.0, 717.0))
        #cam1
        crop_coordinates.append((549.0, 1189.0, 76.0, 716.0))

        self.cams = [vStream(cap_src[i], i, video_res, crop_coordinates[i], rotation_type=cam_rotation_type) for i in range(num_devices)]

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

        if model_type == 'custom':
            # self.model = torch.hub.load(model_directory, model_type, path=model_name, force_reload=True, source=model_source, device='0')
            self.model = torch.hub.load(model_directory, model_type, path="custom_models/tenneco_cam0_N_4.4k.pt", force_reload=True, source=model_source, device='0')
            self.model2 = torch.hub.load(model_directory, model_type, path="custom_models/tenneco_cam1_N_4.4k_speedy.pt", force_reload=True, source=model_source, device='0')
            print("Loaded custom models.")
        else:
            self.model = torch.hub.load(model_directory, model_name, device='0', force_reload=True)
            self.model2 = torch.hub.load(model_directory, model_name, device='0', force_reload=True)
            print(f"Loaded standard models with model_name: {model_name}")

        self.model.eval()
        self.model2.eval()

        # # Load the model
        # self.model = torch.hub.load(model_directory, model_type, path="custom_models/yolov5s.pt", force_reload=True,source=model_source, device='0') \
        #             if model_type == 'custom' else torch.hub.load(model_directory, model_name, device='0', force_reload=True)
        # self.model2 = torch.hub.load(model_directory, model_type, path="custom_models/bestmaskv5.pt", force_reload=True,source=model_source, device='0') \
        #             if model_type == 'custom' else torch.hub.load(model_directory, model_name, device='0', force_reload=True)
        
        # # self.model.classes = [41,65,66,76]  # Set the desired class when using yolov5s for testing
        # self.model.eval() #set the model into eval mode
        # self.model2.eval() #set the model into eval mode

        # Create ByteTracker objects for each camera feed
        self.trackers = [sv.ByteTrack(match_thresh=BYTETRACKER_MATCH_THRESH) for i in range(num_devices)]

        # Set frame params
        self.frame_width = video_res[0]
        self.frame_height = video_res[1]
        self.frame_size = (self.frame_width, self.frame_height)

        self.consecutive_frames_cnt = [0 for i in range(len(self.cams))] # counter for #of frames taken after trigger event
        self.num_consecutive_frames = NUM_CONSECUTIVE_FRAMES # num_consecutive_frames that we want to window (to reduce jitter)
        self.camera_num = 0 # initialize the index of the video stream being processed


        # Set the zone params
        colors = sv.ColorPalette.default()

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
        self.box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.5)
        self.label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)
        # self.blur_annotator = sv.BlurAnnotator()
        # line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
        # line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)



        # Decides if the last 5 seconds should be stored
        if self.record:
            self.recorder = Recorder(system=self)
        


    def ByteTracker_implementation(self, detections):
        # byteTracker is the sv.ByteTrack() object 
        new_detections = []
        if len(detections) != 0:
            new_detections = self.trackers[self.camera_num].update_with_detections(detections)
            #Print both to compare
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
            cam.stopAndRelease()
        print("Released cameras")
        self.jetson.cleanup()
        print("Cleaned up jetson GPIO")
        cv2.destroyAllWindows()
        print("Closed all cv2 windows")
        print("\nExiting...")
        exit(0)

    def save_frames(self, frame, cam_idx=0,  detections=[], save_type='raw' ):
        """
        Save the frames to the disk
        """
        try:
            if save_type == 'raw':
                # Get current date and time in the format YYYYMMDD_HHMMSS
                current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

                cam_dir = Path(os.path.abspath(__file__)).parent / 'saved_frames' / f'cam{cam_idx}'
                cam_dir.mkdir(parents=True, exist_ok=True)
                item_count = get_highest_index(str(cam_dir)) + 1
                file_name = f'img_[{item_count:05d}]_{current_time}'

                img_name = file_name + '.jpg'
                cv2.imwrite(str(cam_dir / img_name), frame)


                if self.save_labels and len(detections):
                    print(f"THESE ARE THE DETCTIONS I'M TRYING TO SAVE: {detections}")
                    label_dir = Path(os.path.abspath(__file__)).parent / 'saved_frames' / f'cam{cam_idx}_labels'
                    label_dir.mkdir(parents=True, exist_ok=True)

                    # write the labels in yolo format to the text file
                    label_name = file_name + '.txt'
                    with open(str(label_dir / label_name), 'w') as file:
                        # for i in range(len(detections[0])):
                        for detection in detections:
                            # Convert from xyxy to yolo format
                            x1, y1, x2, y2 = detection[0]
                            x_center = (x1 + x2) / 2 / frame.shape[1]
                            y_center = (y1 + y2) / 2 / frame.shape[0]
                            width = (x2 - x1) / frame.shape[1]
                            height = (y2 - y1) / frame.shape[0]
                            
                            class_id = detection[-2]
                            print(f"Class id: {class_id}")
                            print(f'writing to file: {class_id} {x_center} {y_center} {width} {height}')
                            
                            file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
            elif save_type == 'annotated':
                # Get current date and time in the format YYYYMMDD_HHMMSS
                current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

                cam_dir = Path(os.path.abspath(__file__)).parent / 'saved_frames' / f'annotated'
                cam_dir.mkdir(parents=True, exist_ok=True)
                item_count = get_highest_index(str(cam_dir)) + 1
                file_name = f'img_[{item_count:05d}]_{current_time}'

                img_name = file_name + '.jpg'
                cv2.imwrite(str(cam_dir / img_name), frame)

        except Exception as e:
            print(f"Error saving frames. Error was:\n {e}")
            traceback.print_exc()

    # @smart_inference_mode()
    def run(self, **kwargs) -> None:
        iou_thres = kwargs.get('iou_thres', 0.7)
        agnostic_nms = kwargs.get('agnostic_nms', False)
        self.annotated_frame = None

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
        print(f"Class names for model 1: {class_names}")
        # class_names = self.model2.module.names if hasattr(self.model2, 'module') else self.model.names
        # print(f"Class names for model 2: {class_names}")

        # Calculate FPS
        frame_count = 0
        start_time = time.time()
        fps=30
        # Initialize second start time varibale to monitor every block of the code:
        # start_time_performance = time.time_ns()
        # Initialize a timer to count the number of seconds since the last detection
        last_detection_time = time.time()
        cam_read_start = time.time()

        while True:
            
            try:

                
                ### Saving a frame from each camera feed ###
                self.captures = []
                self.captures_HD = []
                
                frame_unavailable = False
                avail_cnt=0
                # Check if all cameras have a frame available
                for cam in self.cams:
                    if cam.isFrameAvailable():
                        avail_cnt+=1


                if time.time() - cam_read_start > 5:
                    for cam in self.cams:
                        if not cam.isFrameAvailable():
                            print(f'Camera {cam.src} is not available. Reconnecting...')
                            cam.reconnect()
                    print("resetting cam read")
                    cam_read_start = time.time()
                    continue


                # If yes, store the frames
                if avail_cnt == len(self.cams):
                    cam_read_start = time.time()


                    for cam in self.cams:
                        ret, frame = cam.getFrame()

                        if not ret:
                            frame_unavailable = True
                            break

                        self.captures.append(frame)
                        # self.captures_HD.append(frame_HD)

                    # just in case there's a race condition and the frame is not available
                    if len(self.captures) < len(self.cams):
                        frame_unavailable = True
                # If no, skip to the next iteration
                else:
                    frame_unavailable = True

                if frame_unavailable:
                    continue
                # measure how long reading the frame took
                # print("\n\n Reading frames took: ", (time.time_ns() - start_time_performance) // 1000)
                
                """
                # measure time
                # start_time_performance = time.time_ns()
                # batched_data = torch.stack(self.captures).to('0')  # Stack data into a single batch
                batched_data = torch.stack([torch.tensor(img) for img in self.captures]).to("cuda:0")
                # Print shape:
                # print("Batched data shape: ", batched_data.shape)
                batched_data = batched_data.permute(0, 3, 1, 2)
                # print("Batched data reshaped shape: ", batched_data.shape)
                # measure time
                # print("Stacking frames took: ", (time.time_ns() - start_time_performance) // 1000)
                # start_time_performance = time.time_ns()


                with torch.no_grad():
                    output = self.model(batched_data)
                    predictions = non_max_suppression(output, conf_thres=0.5, iou_thres=0.8)
                # measure time
                # print("Stacked model inference took: ", (time.time_ns() - start_time_performance) // 1000)
                # print(predictions)
                detections = [Detections.empty() for _ in range(len(self.captures))]
                
                """
                ##### Calculate FPS ######
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 5:  # Check every 5 seconds
                    fps = frame_count / elapsed_time
                    if self.show_fps:
                        print(f"FPS: {fps}")
                    # update self.trackers                    
                    # print(f"FPS: {fps}")
                    for cam_num in range(len(self.cams)):
                        self.trackers[cam_num].max_time_lost = round(fps) * DROP_BYTETRACK_AFTER # gives total # of frames to drop lost_tracks after
                    # Reset frame count and time
                    frame_count = 0
                    start_time = time.time()
                ##### Calculate FPS ######




                # If record flag raised, store frames 
                if self.record:
                    self.recorder.store()



                ##### Iterating over frames saved from each of the connected cameras #####
                self.camera_num = 0
                # Iterate over cameras, 1 frame from each  
                for frame in self.captures:
                    # total_time_performance_start = time.time_ns()

                    #Print which camera we are processing
                    # Send through the model
                    # detection_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    detection_frame = frame
                    if self.adjust_brightness:
                        detection_frame = adjust_color(detection_frame)

                    # start_time_performance = time.time_ns()

                    # results = self.model(detection_frame)


                    if not self.data_gather_only:
                        # results = self.model(detection_frame)
                        if self.camera_num == 0:
                            results = self.model(detection_frame)
                        else:
                            results = self.model2(detection_frame)
                        # print(f"Getting results for camera {self.camera_num} in first IF branch")
                    elif self.camera_num == 0:
                        # print(f"Getting results for camera {self.camera_num}")
                        results = self.model(detection_frame)




                    # measure time
                    # print(f"Model inference took for cam{self.camera_num}: {(time.time_ns() - start_time_performance) // 1000}")
                    # start_time_performance = time.time_ns()

                    # Convert the results to a numpy array in the format [x_min, y_min, x_max, y_max, confidence, class]
                    predictions = results.xyxy[0].cpu().numpy()
                    #measeure time
                    # print("Converting to numpy took: ", (time.time_ns() - start_time_performance) // 1000)
                    # start_time_performance = time.time_ns()
                    # Define the confidence threshold
                    conf_thresh = CONFIDENCE_THRESH #0.4

                    # Filter out predictions with a confidence score below the threshold
                    filtered_predictions = predictions[predictions[:, 4] > conf_thresh]
                    # measure time
                    # print("Filtering took: ", (time.time_ns() - start_time_performance) // 1000)
                    # start_time_performance = time.time_ns()

                    # Load the filtered predictions back into the results object
                    results.pred[0] = torch.tensor(filtered_predictions)
                    # measure time
                    # print("Loading back into results took: ", (time.time_ns() - start_time_performance) // 1000)
                    # start_time_performance = time.time_ns()
                    
                    # Convert the detections to the Supervision-compatible format
                    self.detections = sv.Detections.from_yolov5(results)
                    # measure time
                    # print("Converting to Supervision format took: ", (time.time_ns() - start_time_performance) // 1000)
                    # start_time_performance = time.time_ns()
                    

                    if self.use_nms:
                        # Run NMS to remove double detections
                        self.detections = self.detections.with_nms(threshold=iou_thres,  class_agnostic=agnostic_nms)
                        # measure time
                        # print("Running NMS took: ", (time.time_ns() - start_time_performance) // 1000)
                    # start_time_performance = time.time_ns()
                    # If detections present, track them. Assign track_id
                    if len(self.detections) > 0:
                        # Reset the timer if there are detections
                        last_detection_time = time.time()
                        
                        # Update bytetracker with the most recent fps rounded to the nearest integer
                        # self.trackers[self.camera_num].frame_rate = round(fps) #this doesn't work - bytetracker's logic makes no sense

                        # self.detections = self.ByteTracker_implementation(detections=self.detections)
                        # print("###### Detections before ByteTrack: ", self.detections)
                        self.detections = self.trackers[self.camera_num].update_with_detections(self.detections)
                        # print("###### Detections after ByteTrack: ", self.detections)
                    else:
                        # put the detctions through the deafault bytetracker to keep it alive with frame numbers
                        self.detections = self.trackers[self.camera_num].update_with_detections(self.detections)
                        # If there are no detections, check if the timer has exceeded 5 minutes
                        if time.time() - last_detection_time > BYTETRACKER_FULL_RESET_TIMEOUT * 60:
                            # Reinitialize self.trackers if the timer has exceeded 5 minutes
                            # Doing this to make sure that the bytetracker doesn't keep track of detections that are too old & doesn't keep track_ids really high
                            if BaseTrack._count > 1000: #if track_id incrementer reaches 1000
                                BaseTrack._count = 0 # We don't want the track_ids to get out of hand eventually
                            self.trackers = [sv.ByteTrack(match_thresh=BYTETRACKER_MATCH_THRESH) for i in range(len(self.cams))]
                            last_detection_time = time.time()
                    # measure time
                    # print("ByteTracker took: ", (time.time_ns() - start_time_performance) // 1000)
                    # measure time with microsecond precision
                    

        
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
                        self.annotated_frame = frame.copy()
                        labels = [f"#{tracker_id} {self.model.names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, tracker_id in self.detections]
                        # self.annotated_frame = self.label_annotator.annotate(scene=self.annotated_frame, detections=self.detections)
                        self.annotated_frame = self.box_annotator.annotate(scene=self.annotated_frame, detections=self.detections, labels=labels)
                        for zone_annotator, zone in zip(self.zone_annotators, self.zone_polygons):
                            self.annotated_frame = zone_annotator.annotate(scene=self.annotated_frame) if zone.camera_id == self.camera_num else self.annotated_frame
                    
                        # self.annotated_frame = self.zone_annotators[self.camera_num].annotate(scene=self.annotated_frame)

                    # MAYBE USE THIS IN CONJUNCTION WITH SIALOI's CODE
                    # Split into different sets of detections depending on object, by bounding box (aka tuple(goggles/no_goggles, shoes/no_shoes) )
                    # self.item_detections = tuple([self.detections[mask & np.isin(self.detections.class_id, item_sets[i])] for i in range(len(item_sets))])

                    # TRIGGER EVENT - TODO: ADD MULTIPLE TRIGGER EVENTS FUNCTIONALITY
                    if self.trigger_event():
                        # start a object-tied timer using dateime, which we will end after trigger_action is complete
                        self.trigger_action_start_timer = datetime.datetime.now()
                         

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

                            if self.record:
                                # This will be the command for sending in the before and after footage of the violation
                                self.recorder.send()

                            # # Reset the arrays for the data and the images, since we just sent it to the server
                            # self.array_for_frames = [[[] for _ in range(len(self.num_consecutive_frames))] for _ in range(len(self.cams))]
                            # self.detections_array = [[[] for _ in range(len(self.num_consecutive_frames))] for _ in range(len(self.cams))]
                            # self.violations_array = [[[] for _ in range(len(self.num_consecutive_frames))] for _ in range(len(self.cams))]

                            


                    self.other_actions()
                    # Display frame
                    if self.display:
                        fps_text = f"FPS: {fps:.2f}"  # Displaying with 2 decimal points for precision
                        if self.annotate_raw:
                            if self.show_fps:
                                cv2.putText(self.annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                            cv2.imshow(f'Camera {self.cams[self.camera_num].src}', self.annotated_frame)
                        else:
                            if self.show_fps:
                                cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                            cv2.imshow(f'Camera {self.cams[self.camera_num].src}', frame)
                    # Update iteration index for the loop    
                    # print(f"#### Total inference time for cam{self.camera_num}: {(time.time_ns() - total_time_performance_start) // 1000} microseconds")
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