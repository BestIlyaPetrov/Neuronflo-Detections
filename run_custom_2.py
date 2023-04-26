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

auth_token = ""
url = 'http://192.168.0.17:6969/'


CENTER_COORDINATES = (50,50) #Center of the detection region as percentage of FRAME_SIZE
WIDTH = 20 #% of the screen 
HEIGHT = 80 #% of the screen 




def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv5 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[640, 480],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def region_dimensions(frame_size, center, width, height):
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

def authenticate(identifier):

    #INITIAL AUTHENTICATION STAGE
    # identifier = "jetson01"

    # Convert the string to bytes
    string_bytes = identifier.encode('utf-8')

    # Generate the SHA256 hash key
    sha256_key = hashlib.sha256(string_bytes).hexdigest()

    id_data = {
        'identifier':sha256_key
    }

    # Obtain a token
    # response = requests.post(url+'api-token-auth/', data=credentials)
    response = requests.post(url+'device-token-auth', data=id_data)

    if response.status_code == 200:
        token = response.json()['token']
        csrf_token = response.json()['csrf_token']
        # print('Token:', token)
        # print('CSRF Token:', csrf_token)
        return (token, csrf_token)

    else:
        print('Failed to obtain token')
        exit()
    ## END INITIAL AUTHENTICATION



def sendImageToServer(image_bytes, image_data):
    global auth_token
    if auth_token == "":
        auth_token, csrf_token = authenticate(identifier = "jetson01")
        print("New auth token is: ", auth_token)
    else: 
        print("Auth token already is: ", auth_token)


    #ALL SUBSEQUENT DATA POSTS HAPPEN HERE
    # Use the token to authenticate subsequent requests
    headers = {
        'Authorization': f'Token {auth_token}',
        }


    # Get the current time in seconds since the epoch
    timestamp = int(time.time())

    # Convert the timestamp to a string in the dd-mm-yy_hh-mm-ss format
    timestamp_str = time.strftime('%d-%m-%y_%H-%M-%S', time.localtime(timestamp))

    # Print the timestamp string
    # print(timestamp_str)

    response = requests.post(url+'api/entrance_update', files={'image': (timestamp_str+'.jpg', image_bytes)}, data=image_data, headers=headers)


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
        
def main():

    # # Load YOLOv5 model
    #model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    #Load custom YOLOv5 model from file
    model = torch.hub.load('./','custom', path='bestmaskv5.pt', force_reload=True,source='local', device='0')


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
    cap = cv2.VideoCapture(capture_index)  # Use 0 for default camera or provide video file path

    if not cap.isOpened():
        raise ValueError(f"Could not open video device with index {capture_index}")



    # MAIN EXECUTION LOOP
    # save_interval = 10  # Time interval in seconds to save a frame
    # start_time = time.time()
    ret_cnt=0
    try:
        ##INTEGRATING ANYA"S CODE##
        args = parse_argument()

        # cap = cv2.VideoCapture(input_loc)

        #Get frame size
        frame_size = tuple(args.webcam_resolution)
        # Calculate detection region
        zone_polygon = region_dimensions(frame_size, CENTER_COORDINATES, WIDTH, HEIGHT)

        zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=frame_size)
        zone_annotator = sv.PolygonZoneAnnotator(zone=zone,color=sv.Color.blue())
        zone_count=0
        ## END IF INTEGRATION ##

        while True:
            # Read frame from video
            ret, frame = cap.read()
            #Wait a bit before failing (aka if 30 consequtive frames are dropped, then fail)
            if not ret:
                ret_cnt += 1 
                if ret_cnt >= 30:
                    break
                else:
                    continue
            ret_cnt = 0 #resets ret_cnt to 0 if ret == True


            ##INTEGRATING ANYA"S CODE##
            results = model(frame, size=1280)
            detections = sv.Detections.from_yolov5(results)

            # Convert pandas DataFrame to a Python dictionary
            result_dict = results.pandas().xyxy[0].to_dict()

            # Convert dictionary to JSON string
            result_json = json.dumps(result_dict)

            # Print the JSON string
            print(result_json)

            # annotate
            box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

            frame = box_annotator.annotate(scene=frame, detections=detections)
            zone.trigger(detections=detections)

            frame = zone_annotator.annotate(scene=frame)
            


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
                success, encoded_image = cv2.imencode('.jpg', frame)    
                if success:
                    # Convert the encoded image to a byte array
                    image_bytes = bytearray(encoded_image)
                    # You can now use image_data like you did with f.read() 
                    # Send the image to the server
                    sendImageToServer(image_bytes, data)
                    start_time = time.time()
                else:
                    raise ValueError("Could not encode the frame as a JPEG image")



            zone_count = zone.current_count
            #cv2.imshow("bestmaskv5.pt",frame)
            if (cv2.waitKey(30)==27):
                break

            ## END IF INTEGRATION ##



            # # Inference
            # results = model(frame)

            # # Print results to console
            # # print(results.pandas().xyxy[0])

            # # Convert pandas DataFrame to a Python dictionary
            # result_dict = results.pandas().xyxy[0].to_dict()

            # # Convert dictionary to JSON string
            # result_json = json.dumps(result_dict)

            # # Print the JSON string
            # print(result_json)

            # #ADJUST LOGIC HERE - RIGHT NOW IT SENDS A PIC EVERY 10 SEC
            # elapsed_time = time.time() - start_time
            # if elapsed_time > save_interval:
            #     # Encode the image as bytes to send to server
            #     success, encoded_image = cv2.imencode('.jpg', frame)    
            #     if success:
            #         # Convert the encoded image to a byte array
            #         image_bytes = bytearray(encoded_image)
            #         # You can now use image_data like you did with f.read() 
            #         # Send the image to the server
            #         sendImageToServer(image_bytes, result_json)
            #         start_time = time.time()
            #     else:
            #         raise ValueError("Could not encode the frame as a JPEG image")

            # You can now use image_bytes like you did with f.read()

           

    except KeyboardInterrupt:
        # User interrupted execution
        pass

    # Release video capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()