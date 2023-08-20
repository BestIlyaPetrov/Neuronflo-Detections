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
            coordinates[:,0] += i*video_res[0] # shift the x coordinates
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
        
        self.detections_array = []
        self.array_for_frames = []
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
                camera_num = 0
                for frame in self.captures:

                    results = self.model(frame)
                    self.detections = sv.Detections.from_yolov5(results)
                    self.detections = self.detections.with_nms(threshold=iou_thres,  class_agnostic=agnostic_nms)  # apply NMS to detections
                    self.detections = self.ByteTracker_implementation(detections=self.detections, byteTracker=self.trackers[camera_num])

                    if self.annotate:
                        # Annotations
                        mask = []
                        for zone, zone_annotator in zip(self.zones, self.zone_annotators):
                            mask.append(zone.trigger(detections=self.detections)) #this changes self.zones.current_count
                            frame = self.box_annotator.annotate(scene=frame, detections=self.detections)
                            frame = zone_annotator.annotate(scene=frame)

                    # Split into different sets of detections depending on object, by bounding box
                    self.item_detections = tuple([self.detections[mask[i] & np.isin(self.detections.class_id, item_sets[i])] for i in range(len(item_sets))])

                    # TRIGGER EVENT
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
                        cv2.imshow(f'Camera {self.cams[camera_num].src}', frame)

                    # Update iteration index for the loop    
                    camera_num += 1

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
        self.zone_count = 0
        self.cnt = 0

        # If need to overwrite a particular argument do the following. 
        # Let's say need to overwrite the 'model_directory' argument
        # kwargs['model_directory'] = 'my_new_directory_path'

        super().__init__(*args, **kwargs)


    def trigger_event(self) -> bool:

        return self.zones[0].current_count > self.zone_count and (not self.detection_trigger_flag)
    
    def trigger_action(self) -> None:
        self.cnt += 1
        self.detections_array.append(self.item_detections)
        self.array_for_frames.append(self.captures)
        if self.cnt >= self.num_consecutive_frames:
            self.cnt = 0
            self.detection_trigger_flag = False
            the_detections = [[] for _ in range(len(self.items))]
            for detected_items in self.detections_array:
                for the_item in detected_items:
                    if hasattr(the_item, 'class_id') and len(the_item.class_id) > 0:
                        [the_detections[detected_items.index(the_item)].append(int(ids)) for ids in the_item.class_id]
                    
            most_common_detections = self.present_indices
            for i in range(len(self.items)):
                if len(the_detections[i]):
                    most_common_detections[i] = collections.Counter(the_detections[i]).most_common(1)[0][0]
                    print(f"Most common detection for object {self.items[i]} is {most_common_detections[i]}")
                else:
                    print(f"No detections for object {self.items[i]}")
                
            # Pick the least blurry image
            least_blurry_images = [least_blurry_image_indx(self.captures[i]) for i in range(len(self.items))]

            # Compliance Logic
            compliant = False
            if most_common_detections[0] == 0:
                compliant =  False

            elif most_common_detections[0] == 1:
                compliant = True

            bordered_frames = [draw_border(least_blurry_images[i],  compliant, self.border_thickness) for i in range(len(self.items))]
            bordered_frame = np.hstack(tuple(bordered_frames))

            data = {
                'zone_name': '1',
                'crossing_type': 'coming',
                'compliant': str(compliant)
            }

            success, encoded_image = cv2.imencode('.jpg', bordered_frames[0])
            if success:
                image_bytes = bytearray(encoded_image)
                sendImageToServer(image_bytes, data, IP_address=self.server_IP)

                print()
                print("########### DETECTION MADE #############")
  
                print("########### END OF DETECTION #############")
                print()

            else:
                raise ValueError("Could not encode the frame as a JPEG image")
            
            if self.save:
                self.save_frames(self.array_for_frames)

            self.detections_array = []
            self.array_for_frames = []
    
    def other_actions(self) -> None:
        self.zone_count = self.zones[0].current_count
    

class LaserInferenceSystem(InferenceSystem):
    def __init__(self, model_name, video_res, border_thickness, display, save, bboxes, num_devices, model_type, model_directory="./", model_source='local', detected_items=[]) -> None:
        
        super().__init__(model_name, video_res, border_thickness, display, save, bboxes, num_devices, model_type, model_directory, model_source, detected_items)

    def trigger_event(self) -> bool:
        # Trigger event for laser
        return self.zones[0].current_count >= 1
    
    def trigger_action(self) -> None:
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
        super().__init__(model_name, video_res, border_thickness, display, save, bboxes, num_devices, model_type, model_directory, model_source, detected_items)

    def trigger_event(self) -> bool:
        # Trigger event for envision
        return self.zones[0].current_count >= 1
    
    def trigger_action(self) -> None:
        if self.save:
            self.save_frames(self.captures)