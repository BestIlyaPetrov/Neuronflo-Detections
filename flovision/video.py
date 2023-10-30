from threading import Thread, Lock
import cv2
# print("CV2 INFO:", cv2.getBuildInformation())
import numpy as np
import json
import glob, os
from pathlib import Path
import math, time


def upscale_coordinates(x_min, y_min, x_max, y_max, input_resolution, output_resolution):
    # Unpack input and output resolutions
    input_width, input_height   = input_resolution
    output_width, output_height   = output_resolution
    # print 
    print("INPUT RESOLUTION:", input_resolution)
    print("OUTPUT RESOLUTION:", output_resolution)

    # Calculate scaling factors
    width_scale = output_width / input_width
    height_scale = output_height / input_height

    # Upscale the coordinates
    upscaled_x_min = int(round(x_min * width_scale))
    upscaled_y_min = int(round(y_min * height_scale))
    upscaled_x_max = int(round(x_max * width_scale))
    upscaled_y_max = int(round(y_max * height_scale))
    # print both coords


    return upscaled_x_min, upscaled_y_min, upscaled_x_max, upscaled_y_max








def adjust_color(img):

    # Convert to YUV color space
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # Apply CLAHE to the Y channel (luminance)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(4, 4))
    yuv[:,:,0] = clahe.apply(yuv[:,:,0])

    # Convert back to BGR color space
    processed_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    return processed_frame   



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

# def get_device_indices(quantity = 1):
#     # Determine the two sources to use for cameras:
#     # Find all available video devices
#     # devices = glob.glob('/dev/video*')
#     devices = []
#     # port = 554
#     # ip = "192.168.2.51"
#     # devices.append(f'rtsp://{ip}:{port}/stream2')
#     # ip = "192.168.2.50"
#     # devices.append(f'rtsp://{ip}:{port}/stream2')
#     # return devices

#     # Sort the device names in ascending order
#     devices.sort()
#     # Use the first device as the capture index
#     # cap_index = [0,1] #default values aka /dev/video0 and /dev/video1
#     # If there are no devices available, raise an error
#     if not devices:
#         raise ValueError('No video devices found')
#     elif len(devices) < quantity:
#         raise ValueError(f'Not enough cameras connected. Only found {len(devices)}, but need {quantity}')
#     # Otherwise, use the lowest available indexes
#     else:
#         cap_index = []
#         #Creating an array of camera device indices. Aka if we find /dev/video0 and /dev/video2, cap_index == [0,2]
#         for i in range(0, quantity):
#             cap_index.append(int(devices[i][-1]))
#         return cap_index

def draw_border(frame, compliant, border_width=5):
    """
    Draw a red or green border around a given frame.
    """
    if compliant:
        color = (0, 255, 0)  # green
    else:
        color = (0, 0, 255)  # red

    height, width, _ = frame.shape
    bordered_frame = cv2.copyMakeBorder(
        frame, border_width, border_width, border_width, border_width,
        cv2.BORDER_CONSTANT, value=color
    )
    return bordered_frame
    # return bordered_frame[border_width : height + border_width, border_width : width + border_width]


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

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

def least_blurry_image_indx(frame_list):

    blur_val_list = []
    cnt=0
    for frame in frame_list:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        blur_val_list.append(fm)
    # print("BLUR VALS:", blur_val_list)
    return np.array(blur_val_list).argmax()
    # argsort returns indices that would sort the array in ascending order
    # We reverse the order to get indices for descending order of blurriness
    # return np.argsort(blur_val_list)[::-1]



"""
class vStream:
    def __init__(self, src, cam_num, resolution):
        print("Opening camera at link: ", src)
        self.width, self.height = resolution

        # Constructing a GStreamer pipeline for hardware-accelerated decoding (assuming src is an RTSP link)
        gstreamer_pipeline = (
            f'rtspsrc location={src} latency=50 ! rtph264depay ! h264parse ! nvv4l2decoder ! '
            'videoconvert ! video/x-raw, format=(string)BGRx ! '
            f'videoscale ! video/x-raw, width={self.width}, height={self.height} ! appsink'
        )
        # gstreamer_pipeline = (
        #     f'rtspsrc location={src} latency=50 ! rtph265depay ! h265parse ! nvv4l2decoder ! '
        #     'videoconvert ! video/x-raw, format=(string)BGRx ! '
        #     f'videoscale ! video/x-raw, width={self.width}, height={self.height} ! appsink'
        # )
        self.capture = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
        
        # Check if capture was successful
        if not self.capture.isOpened():
            print("Error: Camera not opened")
            print("Error Message:", self.capture.getExceptionMessage())
            raise ValueError("Unable to open camera")

        self.src = cam_num
        self.new_frame_available = False
        self.frame = None
        self.frame_resized = None
        self.lock = Lock()
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
    def update(self):
        while True:
            ret, frame = self.capture.read()
            if not ret:
                print(f"Failed to grab frame from source {self.src}")
                break
            self.lock.acquire()
            self.frame = frame
            self.frame_resized = cv2.resize(frame, (self.width, self.height))
            self.new_frame_available = True
            self.lock.release()

    def getFrame(self):
        self.lock.acquire()
        new_frame = self.new_frame_available
        resized = self.frame_resized
        original = self.frame
        if new_frame:
            self.new_frame_available = False
        self.lock.release()
        return (new_frame, resized, original)
"""



class vStream:
    def __init__(self, src, cam_num, resolution, crop_coordinates, rotation_type=None):
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
        self.crop_coordinates = crop_coordinates

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
        y1, y2, x1, x2 = self.crop_coordinates
        while True:
            if self.running:
                try:
                    getframe = self.capture.read()
                    if self.rotation_type is None:
                        self.frame = getframe[1]
                    else: 
                        self.frame = cv2.rotate(getframe[1],self.rotation_type)

                    # self.frame_resized = cv2.resize(self.frame, (self.inference_width, self.inference_height))
                    
                    self.frame_resized = self.frame[int(y1):int(y2), int(x1):int(x2)]

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
        if self.frame_resized is None:
            return (False, self.frame_resized)
        if self.new_frame_available:
            self.new_frame_available = False
            return (True, self.frame_resized)
        else:
            return (False, self.frame_resized)
    # def getFrame(self):
    #     if self.frame_resized is None:
    #         return (False, self.frame_resized, self.frame)
    #     if self.new_frame_available:
    #         self.new_frame_available = False
    #         return (True, self.frame_resized, self.frame)
    #     else:
    #         return (False, self.frame_resized, self.frame)
        
    def isFrameAvailable(self):
        return self.new_frame_available
    
    def stopAndRelease(self):
        self.running = False
        self.kill_update_loop = True
        self.thread.join()
        self.capture.release()
        


