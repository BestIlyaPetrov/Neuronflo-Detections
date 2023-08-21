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
from .face_id import face_recog

class InferenceSystem:
    """
    General class for inference system
    """
    def __init__(self, model_name, video_res, border_thickness, display, save, bboxes, num_devices, model_type='custom', model_directory="./", model_source='local', detected_items=[], server_IP='local', annotate=False) -> None:
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
            # coordinates[:,0] += i*video_res[0] # shift the x coordinates
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
        
        # self.detections_array = []
        # self.array_for_frames = []
        self.detections_array = [[] for _ in range(len(self.zones))]

        self.array_for_frames = [[] for _ in range(len(self.zones))]

        cnt = 0
        self.detection_trigger_flag = False

        # Split detections into different sets depending on object, by bounding box (aka [goggles, no_goggles])
        self.present_indices = [2* idx for idx in range(len(self.items))]
        self.absent_indices = [2* idx + 1 for idx in range(len(self.items))]
        item_sets = [[present, absent] for present, absent in zip(self.present_indices, self.absent_indices)]

        while True:
            try:
                # Main detection loop
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
              
                # frame = np.hstack(tuple(self.captures))

                #Iterating over frames for each of the connected cameras
                # TODO: ENSURE THIS WORKS FOR SEVERAL CAMERA STREAMS
                
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

                    
                    # Check # of detections in a zone (We are assuming there's 1 zone per camera - TODO: UPGRADE TO MULTIPLE)
                    mask = self.zones[self.camera_num].trigger(detections=self.detections) #this changes self.zones.current_count
                    
                    # Annotate the zones and the detections on the frame if the flag is set
                    if self.annotate:
                        frame = self.box_annotator.annotate(scene=frame, detections=self.detections)
                        frame = self.zone_annotators[self.camera_num].annotate(scene=frame)

                    # Split into different sets of detections depending on object, by bounding box (aka tuple(goggles/no_goggles, shoes/no_shoes) )
                    self.item_detections = tuple([self.detections[mask & np.isin(self.detections.class_id, item_sets[i])] for i in range(len(item_sets))])

                    # TRIGGER EVENT - TODO: ADD MULTIPLE TRIGGER EVENTS FUNCTIONALITY
                    if self.trigger_event():
                        self.detection_trigger_flag = True
                        
                        results_dict = results.pandas().xyxy[0].to_dict()
                        results_json = json.dumps(results_dict)
                        print()
                        print("RESULTS: ", results)
                        print("RESULTS JSON: ", results_json)
                        print()

                    if self.detection_trigger_flag:
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


class EntranceInferenceSystem(InferenceSystem):
    def __init__(self, *args, **kwargs) -> None:


        # If need to overwrite a particular argument do the following. 
        # Let's say need to overwrite the 'model_directory' argument
        # kwargs['model_directory'] = 'my_new_directory_path'

        super().__init__(*args, **kwargs)
        self.zone_count = [0 for i in range(len(self.cams))]
        self.consecutive_frames_cnt = [0 for i in range(len(self.cams))]


    def trigger_event(self) -> bool:

        return int(self.zones[0].current_count) > self.zone_count[0] and (not self.detection_trigger_flag) #FIX THIS FLAG LOGIC
    
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
            self.detection_trigger_flag = False

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
        results = self.model(frame)

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
    
class FaceRecognitionSystem(InferenceSystem):
    def __init__(self, model_name, video_res, border_thickness, display, save, bboxes, num_devices, model_type, model_directory="./", model_source='local', detected_items=[]) -> None:
        
        super().__init__(model_name, video_res, border_thickness, display, save, bboxes, num_devices, model_type, model_directory, model_source, detected_items)

    def trigger_event(self) -> bool:
        # Trigger event for face recognition
        return self.zones[0].current_count >= 1
    
    def trigger_action(self) -> None:
        if self.save:
            self.save_frames(self.captures)

class EnvisionInferenceSystem(InferenceSystem):
    def __init__(self, model_name, video_res, border_thickness, display, save, bboxes, num_devices, model_type='custom', model_directory="./", model_source='local', detected_items=[]) -> None:
        self.detection_trigger_flag = False
        super().__init__(model_name, video_res, border_thickness, display, save, bboxes, num_devices, model_type, model_directory, model_source, detected_items)
    
    
    
    def run(self, iou_thres, agnostic_nms):
        print("Inference successfully launched")
        self.detection_trigger_flag = False
        byte_tracker = sv.ByteTracker()
        while True:
            try:
                # Make the slowest cam be the bottleneck here
                ret, frame = self.cams[0].getFrame()
                ret2, frame2 = self.cams[1].getFrame()
                if ret == False or ret2 == False:
                    continue

                # Run Inference
                results = self.model(frame)
                results2 = self.model(frame2)

                # load results into supervision
                detections = sv.Detections.from_yolov5(results)
                detections2 = sv.Detections.from_yolov5(results2)
                
                # Apply NMS to remove double detections
                detections = detections.with_nms(threshold=iou_thres,  class_agnostic=agnostic_nms)
                detections2 = detections2.with_nms(threshold=iou_thres,  class_agnostic=agnostic_nms)
                
                # Check if the distance metric has been violated
                self.detection_trigger_flag, detection_info = self.trigger_event(detections=detections)
                # The detection_info variable will hold a tuple that
                # will be the indices for the objects in violation
                
                # Gives all detections ids and will be processed in the next step
                detections = self.ByteTracker_implementation(detections=detections, byteTracker=byte_tracker)
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
                tracker_id = detections.tracker_id[detection_info[1]]



                #######################################################################
                # Place logic for detecting if the time passed for an ID's violation  #
                # is under 10 mins. If so, then no violation has occurred. However,   #
                # if it's been over 10 mins. Make sure that the function will take
                # in the tracker id.                                     #
                #######################################################################
                # is under 10 mins. If so, then no violation has occurred. However,   #
                # if it's been over 10 mins. Make sure that the function will take
                # in the tracker id.                                     #
                #######################################################################


                # Display frame
                if self.display:
                    cv2.imshow('ComboCam', frame)

            except Exception as e:
                print('frame unavailable', e)
                traceback.print_exc()

            if cv2.waitKey(1) == ord('q'):
                self.stop()
    
    def findCenter(self, minX, minY, maxX, maxY):
        centerX = round(minX + (maxX - minX)/2)
        centerY = round(minY + (maxY - minY)/2)
        return centerX, centerY

    def trigger_event(self, detections) -> bool:
                 # Trigger event for envision
        # Variables in the for loop to be processed
        labels = detections.labels
        cnt = 0
        violation_detected = False
        threshold = 0.3
        # Change the class numbers for the labels to the correct one
        # Change the class numbers for the labels to the correct one
        for label in labels:
            if label == 0: # Assuming class 0 is "soldering"
                minX, minY, maxX, maxY = detections.boxes[cnt]
                centerX, centerY = self.findCenter(minX=minX, minY=minY, maxX=maxX, maxY=maxY)
                cnt2 = 0
                for label2 in labels:
                    if label2 == 1: # Assuming class 1 is "no_goggles"
                        minX2, minY2, maxX2, maxY2 = detections.boxes[cnt2]
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

        if self.save:
            self.save_frames(self.captures)