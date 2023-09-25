import cv2
import torch
import numpy as np
import os
from pathlib import Path
import traceback
import supervision as sv
from datetime import datetime
from .video import vStream, get_device_indices
from .utils import get_highest_index, findLocalServer
from .jetson import Jetson
from .comms import sendImageToServer
from .bbox_gui import create_bounding_boxes, load_bounding_boxes


class InferenceModel:
    def __init__(self, model_directory, model_type, model_name, model_source='local', device='0'):
        self.model = self.load_model(model_directory, model_type, model_name, model_source, device)
        self.model.eval()
        
    @staticmethod
    def load_model(model_directory, model_type, model_name, model_source, device):
        if model_type == 'custom':
            return torch.hub.load(model_directory, model_type, path=model_name, force_reload=True, source=model_source, device=device)
        else:
            return torch.hub.load(model_directory, model_name, device=device, force_reload=True)
        

class ByteTracker:
    def __init__(self, match_thresh=0.4):
        self.tracker = sv.ByteTrack(match_thresh)

    def update_with_detections(self, detections):
        if len(detections) != 0:
            new_detections = self.tracker.update_with_detections(detections)
            return self.update_class_ids(new_detections, detections)
        return []

    @staticmethod
    def update_class_ids(new_detections, detections):
        sorted_original_indices = np.argsort(detections.confidence)
        sorted_new_indices = np.argsort(new_detections.confidence)
        index_mapping = dict(zip(sorted_new_indices, sorted_original_indices))
        for new_idx, original_idx in index_mapping.items():
            new_detections.class_id[new_idx] = detections.class_id[original_idx]
        return new_detections


class Camera:
    def __init__(self, src, resolution):
        self.capture = vStream(src, resolution)
        self.src = src

    def get_frame(self):
        return self.capture.getFrame()

class Zone:
    def __init__(self, zone_id, frame_resolution_wh, polygon, color, thickness=1, text_thickness=1, text_scale=1):
        self.zone_id = zone_id
        self.zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=frame_resolution_wh)
        self.annotator = sv.PolygonZoneAnnotator(zone=self.zone, color=color, thickness=thickness, text_thickness=text_thickness, text_scale=text_scale)

    def trigger(self, detections):
        return self.zone.trigger(detections=detections)
    
    def check_and_store_violations(self, detections, camera_idx):
        triggered_detections = self.trigger(detections)
        violations = []
        for detection in triggered_detections:
            violation = Violation(camera_id=camera_idx, class_id=detection.class_id, timestamp=datetime.datetime.now(), violation_code=f"Zone{self.zone_id}")
            violations.append(violation)
        return violations


class InferenceSystem:
    def __init__(self, **kwargs):
        self._initialize_attributes(**kwargs)
        self._initialize_cameras()
        self._initialize_zones()
        self._initialize_directories()
        self.model = InferenceModel(self.model_directory, self.model_type, self.model_name, self.model_source).model
        self.trackers = [ByteTracker() for _ in range(self.num_devices)]
        self.box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=1)
        self.violations_dict = {}  # Dictionary to store violations with track_id as key

    def _initialize_attributes(self, **kwargs):
        self.model_name = kwargs.get('model_name')
        self.video_res = kwargs.get('video_res', (640, 480))
        self.border_thickness = kwargs.get('border_thickness', 1)
        self.display = kwargs.get('display', False)
        self.save = kwargs.get('save', False)
        self.bboxes = kwargs.get('bboxes', False)
        self.num_devices = kwargs.get('num_devices', 1)
        self.model_type = kwargs.get('model_type', 'custom')
        self.model_directory = kwargs.get('model_directory', "./")
        self.model_source = kwargs.get('model_source', 'local')
        self.detected_items = kwargs.get('detected_items', [])
        self.server_IP = kwargs.get('server_IP', findLocalServer())
        self.annotate_raw = kwargs.get('annotate_raw', False)
        self.annotate_violation = kwargs.get('annotate_violation', False)
        self.debug = kwargs.get('debug', False)
        self.cams = []
        self.zones = []
        self.item_dirs = []
        self.consecutive_frames_cnt = [0 for _ in range(self.num_devices)]
        self.jetson = Jetson()
        
    def _initialize_cameras(self):
        cap_indices = get_device_indices(quantity=self.num_devices)
        self.cams = [Camera(cap_indices[i], self.video_res) for i in range(self.num_devices)]

    def _initialize_zones(self):
        func = create_bounding_boxes if self.bboxes else load_bounding_boxes
        zone_polygons_per_cam = [func(cam.capture) for cam in self.cams]
        colors = sv.ColorPalette.default()
        for i, zone_polygons in enumerate(zone_polygons_per_cam):
            self.zones.append([Zone(j, self.video_res, np.array(polygon), colors.by_idx(j)) for j, polygon in enumerate(zone_polygons)])

        
    def _initialize_directories(self):
        self.item_dirs = [Path(f"./data/{item}") for item in self.detected_items]
        for dir_ in self.item_dirs:
            dir_.mkdir(exist_ok=True, parents=True)

    def save_detections(self, idx, detections):
        max_idx = get_highest_index(self.item_dirs[idx])
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        image_file_path = self.item_dirs[idx] / f"{timestamp}_{max_idx}.png"
        cv2.imwrite(str(image_file_path), detections.frame)
        sendImageToServer(image_file_path, self.server_IP)

    def infer(self):
        for idx, cam in enumerate(self.cams):
            try:
                frame = cam.get_frame()
                detections = self.model.detect(frame, self.detected_items)
                detections = self.trackers[idx].update_with_detections(detections)
                self._process_detections(idx, detections)
            except Exception as e:
                if self.debug:
                    print("Error occurred: ", e)
                    traceback.print_exc()


    def _display_or_save(self, idx, detections):
        if self.display:
            cv2.imshow(f"Camera {idx}", detections.frame)
        if self.save:
            save_path = Path(f"./data/cam_{idx}")
            save_path.mkdir(exist_ok=True, parents=True)
            filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            cv2.imwrite(str(save_path / f"{filename}.png"), detections.frame)

    def release_resources(self):
        for cam in self.cams:
            cam.capture.release()
        cv2.destroyAllWindows()

    def _process_detections(self, idx, detections):
        violations = []
        for zone in self.zones[idx]:
            zone_violations = zone.check_and_store_violations(detections, idx)
            violations.extend(zone_violations)
        
        if violations:
            self.handle_violations(violations, idx, detections)
        
        if self.save or self.display:
            frame = detections.frame.copy()
            for zone in self.zones[idx]:
                frame = zone.annotator.annotate(frame)
            detections.frame = frame
            self._display_or_save(idx, detections)

        if self.annotate_raw:
            detections = self.box_annotator.annotate(detections)
            self._display_or_save(idx, detections)

    def handle_violations(self, violations, idx, detections):
        timestamp = datetime.datetime.now()
        for violation in violations:
            track_id = violation.detection.track_id  # Assume detection has a track_id attribute
            camera_id = idx
            violation_code = f"Zone{violation.zone_id}"  # Example, you can modify as per requirement
            class_id = violation.detection.class_id  # Assume detection has a class_id attribute

            if track_id in self.violations_dict:
                violation_obj = self.violations_dict[track_id]
                if not violation_obj.Check_Code(violation_code, class_id):
                    violation_obj.Add_Code(violation_code, class_id)
            else:
                violation_obj = Violation(camera_id, class_id, timestamp, violation_code)
                self.violations_dict[track_id] = violation_obj

            print(f"Violation detected in Zone {violation.zone_id} of Camera {camera_id}, Track ID: {track_id}")

        self.prune_violations()

    def prune_violations(self):
        to_remove = []
        for track_id, violation_obj in self.violations_dict.items():
            if not violation_obj.Update_Time():
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.violations_dict[track_id]

    def save_detections(self, idx, detections, violations):
        max_idx = get_highest_index(self.item_dirs[idx])
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        image_file_path = self.item_dirs[idx] / f"{timestamp}_{max_idx}.png"
        cv2.imwrite(str(image_file_path), detections.frame)
        sendImageToServer(image_file_path, self.server_IP)


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
        condition = [((datetime.datetime.now() - timestamp) < datetime.timedelta(minutes=1)) for timestamp in self.timestamps]
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
