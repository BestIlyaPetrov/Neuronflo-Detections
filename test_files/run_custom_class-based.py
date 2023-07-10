import cv2
import torch
import glob
import requests
import os
import time
import json
import numpy as np
import io
import hashlib

import supervision as sv
import argparse
import operator


class YOLOv5Detector:
    def __init__(self, capture_index=0, model_path='../trained_models/bestmaskv5.pt', input_res=(1920,1080) ):

        # Load YOLOv5 model
        self.model = torch.hub.load('./', 'custom', path=model_path, force_reload=True, source='local', device='0')
        self.input_res = input_res
        # Find all available video devices
        devices = glob.glob('/dev/video*')

        # Sort the device names in ascending order
        devices.sort()

        # If there are no devices available, raise an error
        if not devices:
            raise ValueError('No video devices found')

        # Otherwise, use the lowest available index
        else:
            self.capture_index = capture_index if capture_index is not None else int(devices[0][-1])

        # Open video capture
        self.cap = cv2.VideoCapture(self.capture_index)  # Use 0 for default camera or provide video file path
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_res[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_res[1])

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video device with index {self.capture_index}")
    
    def region_dimensions(self, frame_size, center, width, height):
        # Calculate the width and height of the detection region in pixels
        region_width = int(frame_size[0] * width / 100)
        region_height = int(frame_size[1] * height / 100)

        # Calculate the coordinates of the top-left corner of the detection region
        region_x = int((center[0] / 100) * frame_size[0] - (region_width / 2))
        region_y = int((center[1] / 100) * frame_size[1] - (region_height / 2))

        # Calculate the coordinates of the four corners of the detection region
        top_left = [region_x, region_y]
        top_right = [region_x + region_width, region_y]
        bottom_right = [region_x + region_width, region_y + region_height]
        bottom_left = [region_x, region_y + region_height]

        # Create a numpy array of the corner coordinates
        zone_polygon = np.array([top_left, top_right, bottom_right, bottom_left])

        return zone_polygon

    def run_detection(self, center_coordinates, width, height):
        # Get frame size
        # frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # Calculate detection region
        zone_polygon = self.region_dimensions(self.input_res, center_coordinates, width, height)

        zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=self.input_res)
        zone_annotator = sv.PolygonZoneAnnotator(zone=zone,color=sv.Color.blue())
        zone_count = 0
        try:
            # MAIN EXECUTION LOOP
            ret_cnt = 0
            while True:
                # Read frame from video
                ret, self.frame = self.cap.read()
                # Wait a bit before failing (aka if 30 consecutive frames are dropped, then fail)
                if not ret:
                    ret_cnt += 1 
                    if ret_cnt >= 30:
                        break
                    else:
                        continue
                ret_cnt = 0 #resets ret_cnt to 0 if ret == True

                # Perform object detection
                results = self.model(self.frame)
                detections = sv.Detections.from_yolov5(results)

                # Convert pandas DataFrame to a Python dictionary
                result_dict = results.pandas().xyxy[0].to_dict()

                # Convert dictionary to JSON string
                result_json = json.dumps(result_dict)

                # Print the JSON string
                print(result_json)


                # annotate
                box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

                self.frame = box_annotator.annotate(scene=self.frame, detections=detections)
                zone.trigger(detections=detections)

                self.frame = zone_annotator.annotate(scene=self.frame)
                cv2.imshow("../trained_models/bestmaskv5.pt",self.frame)

                if(operator.and_(not(zone_count),zone.current_count)):

                    #Get data ready
                    compliant=False
                    if(detections.class_id.any() == 0):
                        compliant=False #mask
                    elif(detections.class_id.all() == 1):
                        compliant=False #no_mask

                    data = {
                            'zone_name': '1',
                            'crossing_type': 'coming',
                            'compliant' : str(compliant)
                        }
                    #convert image to be ready to be sent
                    success, encoded_image = cv2.imencode('.jpg', self.frame)    
                    if success:
                        # Convert the encoded image to a byte array
                        image_bytes = bytearray(encoded_image)
                        # You can now use image_data like you did with f.read() 

                        # Send the image to the server
                        
                        # sendImageToServer(image_bytes, data)
                    else:
                        raise ValueError("Could not encode the frame as a JPEG image")



                    zone_count = zone.current_count
                    # cv2.imshow("../trained_models/bestmaskv5.pt",frame)
                    

                ## END OF INTEGRATION ##
        except KeyboardInterrupt:
            if (cv2.waitKey(30)==27):
                            # Release video capture and destroy any OpenCV windows
                self.cap.release()
                cv2.destroyAllWindows()
                
 
            




detector = YOLOv5Detector( model_path='../trained_models/bestmaskv5.pt', input_res=(1920,1080))
detector.run_detection(center_coordinates=(50, 50), width=50, height=50)
