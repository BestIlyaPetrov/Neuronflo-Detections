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
import threading

class YOLOv5Live:
    def __init__(self, center_coordinates, width, height, url, identifier):
        self.CENTER_COORDINATES = center_coordinates
        self.WIDTH = width
        self.HEIGHT = height
        self.url = url
        self.identifier = identifier
        self.auth_token = ""

    def parse_argument(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="YOLOv5 live")
        parser.add_argument(
            "--webcam-resolution",
            default=[640, 480],
            nargs=2,
            type=int
        )
        args = parser.parse_args()
        return args

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

    def authenticate(self):

        #INITIAL AUTHENTICATION STAGE

        # Convert the string to bytes
        string_bytes = self.identifier.encode('utf-8')

        # Generate the SHA256 hash key
        sha256_key = hashlib.sha256(string_bytes).hexdigest()

        id_data = {
            'identifier':sha256_key
        }

        # Obtain a token
        response = requests.post(self.url+'device-token-auth', data=id_data)

        if response.status_code == 200:
            token = response.json()['token']
            csrf_token = response.json()['csrf_token']
            return (token, csrf_token)

        else:
            print('Failed to obtain token')
            exit()
        ## END INITIAL AUTHENTICATION

    def sendImageToServer(self, image_bytes, image_data):
        if self.auth_token == "":
            self.auth_token, csrf_token = self.authenticate()
            print("New auth token is: ", self.auth_token)
        else: 
            print("Auth token already is: ", self.auth_token)

        #ALL SUBSEQUENT DATA POSTS HAPPEN HERE
        # Use the token to authenticate subsequent requests
        headers = {
            'Authorization': f'Token {self.auth_token}',
            }

        # Get the current time in seconds since the epoch
        timestamp = int(time.time())

        # Convert the timestamp to a string in the dd-mm-yy_hh-mm-ss format
        timestamp_str = time.strftime('%d-%m-%y_%H-%M-%S', time.localtime(timestamp))

        response = requests.post(self.url+'api/entrance_update', files={'image': (timestamp_str+'.jpg', image_bytes)}, data=image_data, headers=headers)

        # Check response status code
        if response.status_code == 200:
            try: 
                msg = response.json()['message']
                print(msg)
            except Exception as e:
                print(e)
        else:
            print('Failed to upload image - HTTP response status: ',response.status_code )
            try: 
                msg = response.json()['message']
                print(msg)
            except Exception as e:
                print(e)

    def main(self):

        # Load custom YOLOv5 model from file
        model = torch.hub.load('./','custom', path='bestmaskv5.pt', force_reload=True, source='local', device='0')

        # Find all available video devices
        devices = glob.glob('/dev/video*')

        # Sort the device names in ascending order
        devices.sort()

        # Use the first device as the capture index
        capture_index = 0

        # If there are no devices available, raise an error
        if not devices:
            raise ValueError('No video devices found')

        # Otherwise, use the lowest available index
        else:
            capture_index = int(devices[0][-1])

        # Open video capture
        input_res= (1920,1080)
        cap = cv2.VideoCapture(capture_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_res[1])

        if not cap.isOpened():
            raise ValueError(f"Could not open video device with index {capture_index}")

        # MAIN EXECUTION LOOP
        ret_cnt=0
        try:
            args = self.parse_argument()

            # Get frame size
            frame_size = input_res

            # Calculate detection region
            zone_polygon = self.region_dimensions(frame_size, self.CENTER_COORDINATES, self.WIDTH, self.HEIGHT)

            zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=frame_size)
            zone_annotator = sv.PolygonZoneAnnotator(zone=zone,color=sv.Color.blue())
            zone_count=0

            while True:
                # Read frame from video
                ret, frame = cap.read()

                if not ret:
                    ret_cnt += 1 
                    if ret_cnt >= 30:
                        break
                    else:
                        continue
                ret_cnt = 0

                results = model(frame)
                detections = sv.Detections.from_yolov5(results)

                result_dict = results.pandas().xyxy[0].to_dict()
                result_json = json.dumps(result_dict)
                print(result_json)

                box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
                frame = box_annotator.annotate(scene=frame, detections=detections)
                zone.trigger(detections=detections)
                frame = zone_annotator.annotate(scene=frame)

                if(operator.and_(not(zone_count),zone.current_count)):
                    compliant=False
                    if(detections.class_id.any() == 0):
                        compliant=False
                    elif(detections.class_id.all() == 1):
                        compliant=False

                    data = {
                            'zone_name': '1',
                            'crossing_type': 'coming',
                            'compliant' : str(compliant)
                        }

                    success, encoded_image = cv2.imencode('.jpg', frame)
                    if success:
                        image_bytes = bytearray(encoded_image)
                        self.sendImageToServer(image_bytes, data)
                        start_time = time.time()
                    else:
                        raise ValueError("Could not encode the frame as a JPEG image")

                zone_count = zone.current_count

                if (cv2.waitKey(30) == 27):
                    break

        except KeyboardInterrupt:
            # User interrupted execution
            pass

        # Release video capture and destroy any OpenCV windows
        cap.release()
        cv2.destroyAllWindows()



def run_detector_instance():
    detector = MaskDetector()
    detector.main()

if __name__ == "__main__":
    # Create two threads to run the MaskDetector instances
    thread1 = threading.Thread(target=run_detector_instance)
    thread2 = threading.Thread(target=run_detector_instance)

    # Start the threads
    thread1.start()
    thread2.start()

    # Wait for both threads to finish
    thread1.join()
    thread2.join()
