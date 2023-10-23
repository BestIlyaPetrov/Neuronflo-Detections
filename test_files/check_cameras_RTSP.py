
from threading import Thread, Lock
import cv2
# print("CV2 INFO:", cv2.getBuildInformation())
import numpy as np
import json
import glob, os
from pathlib import Path
import math, time
import argparse
import traceback





"""
Example config.json file:

{
    "cameras": [
      {
        "type":"rtsp",
        "ip": "192.168.2.51",
        "port": 554,
        "path": "/stream1"
      },
      {
        "type":"rtsp",
        "ip": "192.168.2.50",
        "port": 554,
        "path": "/stream1"
      }
    ]
  }


 or (TBDownloaded from tenneco jetson)


"""
def get_camera_src(quantity=1):
# Get the full path of the current script
    script_path = os.path.abspath(__file__)

    # Get the directory containing the current script
    script_directory = os.path.dirname(script_path)
    devices = []
    with open(Path(script_directory) / 'video-config.json', 'r') as f:
        config = json.load(f)

    for camera in config["cameras"]:
        if camera["type"] == "rtsp":
            ip = camera.get("ip")
            username = camera.get("username")
            password = camera.get("password")
            port = camera.get("port", 554)  # Default to 554 if port is not provided
            path = camera.get("path", "")  # Default to empty string if path is not provided

            # Check if channelNo and typeNo are provided, and format the path accordingly
            if "channelNo" in camera and "typeNo" in camera:
                channelNo = camera["channelNo"]
                typeNo = camera["typeNo"]
                path = path.format(channelNo=channelNo, typeNo=typeNo)

            # Construct video URL based on the presence of username and password
            if username and password:
                video_url = f'rtsp://{username}:{password}@{ip}:{port}{path}'
            else:
                video_url = f'rtsp://{ip}:{port}{path}'

            devices.append(video_url)
        elif camera["type"] == "usb":
            usb_devices = glob.glob('/dev/video*')
            if not usb_devices:
                raise ValueError('No usb video devices found')
            usb_devices.sort()
            for cam in usb_devices:
                if cam[-1] in devices:
                    continue
                else:
                    devices.append(cam[-1])
                    break
            


    return devices


class vStream:
    def __init__(self, src, cam_num, resolution, rotation_type=None):
        print("Opening camera at link: ", src)
        self.rotation_type = rotation_type
        self.inference_width = resolution[0]
        self.inference_height = resolution[1]
        self.src_link = src
        self.src = cam_num
        self.new_frame_available = False
        self.frame = None
        self.frame_resized= None
        self.running = False
        self.kill_update_loop = False

        if self.rotation_type not in [cv2.ROTATE_90_CLOCKWISE,cv2.ROTATE_90_COUNTERCLOCKWISE,cv2.ROTATE_180]:
            self.rotation_type = None
            print("Rotation type not supported. Defaulting to no rotation")
        
        self.capture=cv2.VideoCapture(src)
        if self.capture.isOpened():
            self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.capture.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.fps = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback
            print(f'Success frames {self.width}x{self.height} at {self.fps:.2f} FPS')

            self.thread = Thread(target=self.update, args=())
            self.thread.daemon=True
            self.running = True
            self.thread.start()
            print(f"Started thread for cam {cam_num}")
        else:
            self.reconnect()
        
    def reconnect(self):

        try:
            """Attempt to reconnect the camera."""
            print(f"Attempting to reconnect cam {self.src} at {self.src_link}...")
            self.running = False
            time.sleep(1)  # give it a short break before reconnecting
            self.capture = cv2.VideoCapture(self.src_link)
            if self.capture.isOpened():
                self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = self.capture.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
                self.fps = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback
                print(f'Success frames {self.width}x{self.height} at {self.fps:.2f} FPS')
                self.running = True
            else:
                raise ValueError(f"Unable to open camera {self.src} at {self.src_link}")
        except Exception as e:
            print(f"Can't reconnect because {e}")
            self.reconnect()


    def update(self):
        fps_cnt = 0 
        cnt=0
        start_time = time.time()
        while True:
            if self.running:
                try:
                    getframe = self.capture.read()
                    if self.rotation_type is None:
                        self.frame = getframe[1]
                    else: 
                        self.frame = cv2.rotate(getframe[1],self.rotation_type)

                    self.frame_resized = cv2.resize(self.frame, (self.inference_width, self.inference_height))
                    self.new_frame_available = True
                    # fps_cnt += 1
                    # if cnt % 10 == 0:
                    #     elapsed_time = time.time() - start_time
                    #     fps = fps_cnt / elapsed_time
                    #     print(f"Actual camera FPS: {fps:.2f} for cam {self.src}")
                    #     fps_cnt = 0
                    #     start_time = time.time()
                    if cnt > 0:
                        print(f"Stream from cam {self.src} is available again")
                        cnt = 0
                except:
                    if cnt < 1:
                        print(f"Frames can't be read from cam {self.src}")
                        cnt +=1
                        self.reconnect()
                    continue
            if self.kill_update_loop:
                break


    def getFrame(self):
        # print(f"Frame available for cam {self.src} is {self.new_frame_available}")
        if self.frame_resized is None:
            return (False, self.frame_resized, self.frame)
        if self.new_frame_available:
            self.new_frame_available = False
            return (True, self.frame_resized, self.frame)
        else:
            return (False, self.frame_resized, self.frame)
        
    def isFrameAvailable(self):
        return self.new_frame_available
    
    def stopAndRelease(self):
        self.running = False
        self.kill_update_loop = True
        self.thread.join()
        self.capture.release()
        





def main(HD = False):

    cam_rotation_type = cv2.ROTATE_90_COUNTERCLOCKWISE

    video_res = [384, 640]

    cap_src = get_camera_src(quantity = 2)

    cams = [vStream(cap_src[i], i, video_res, rotation_type=cam_rotation_type) for i in range(2)]


    cam_read_start = time.time()
    start_time = time.time()
    launch_time = time.time()
    frame_count =0
    fps = 30
    while True:
        try: 
            captures = []
            captures_HD = []
            
            frame_unavailable = False
            avail_cnt=0
            # Check if all cameras have a frame available
            for cam in cams:
                if cam.isFrameAvailable():
                    avail_cnt+=1


            if time.time() - cam_read_start > 5:
                for cam in cams:
                    if not cam.isFrameAvailable():
                        print(f'Camera {cam.src} is not available. Reconnecting...')
                        cam.reconnect()
                print("resetting cam read")
                cam_read_start = time.time()
                continue

            # If yes, store the frames
            if avail_cnt == len(cams):
                cam_read_start = time.time()


                for cam in cams:
                    ret, frame, frame_HD = cam.getFrame()


                    if not ret:
                        frame_unavailable = True
                        break

                    captures.append(frame)
                    captures_HD.append(frame_HD)
                # just in case there's a race condition and the frame is not available
                if len(captures) < len(cams):
                    frame_unavailable = True
            # If no, skip to the next iteration
            else:
                frame_unavailable = True

            if frame_unavailable:
                continue
    
            ##### Calculate FPS ######
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 5:  # Check every 5 seconds
                fps = frame_count / elapsed_time
                print(f"FPS: {fps}")
                # Reset frame count and time
                frame_count = 0
                start_time = time.time()
            ##### Calculate FPS ######
            #add a pause before displaying the frames
            
            for camera_num in range(len(cams)):
                if HD:
                    frame_to_display = captures_HD[camera_num]
                else:
                    frame_to_display = captures[camera_num]

                fps_text = f"FPS: {fps:.2f}"  # Displaying with 2 decimal points for precision
                cv2.putText(frame_to_display, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(f'Camera {camera_num}', frame_to_display)
                # cv2.waitKey(1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopping detections and releasing cameras")
                for cam in cams:
                    cam.stopAndRelease()
                print("Released cameras")
                exit(0)

        except Exception as e:
            print("Error in main loop:", e)
            traceback.print_exc()
            break

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--HD', action='store_true', help='show HD video feed')

    options = parser.parse_args()
    return options


if __name__ == "__main__":
    opt = parse_options()
    main(**vars(opt))



