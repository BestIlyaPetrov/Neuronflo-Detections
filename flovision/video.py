from threading import Thread
import cv2
import numpy as np
import json
import glob





def adjust_color(img):

    # Convert to YUV color space
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # Apply CLAHE to the Y channel (luminance)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(4, 4))
    yuv[:,:,0] = clahe.apply(yuv[:,:,0])

    # Convert back to BGR color space
    processed_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    return processed_frame   


def get_device_indices(quantity = 1):
    # Determine the two sources to use for cameras:
    # Find all available video devices
    # devices = glob.glob('/dev/video*')
    devices = []
    port = 554
    ip = "192.168.2.51"
    devices.append(f'rtsp://{ip}:{port}/stream2')
    ip = "192.168.2.50"
    devices.append(f'rtsp://{ip}:{port}/stream2')
    return devices
    # # Sort the device names in ascending order
    # devices.sort()
    # # Use the first device as the capture index
    # # cap_index = [0,1] #default values aka /dev/video0 and /dev/video1
    # # If there are no devices available, raise an error
    # if not devices:
    #     raise ValueError('No video devices found')
    # elif len(devices) < quantity:
    #     raise ValueError(f'Not enough cameras connected. Only found {len(devices)}, but need {quantity}')
    # # Otherwise, use the lowest available indexes
    # else:
    #     cap_index = []
    #     #Creating an array of camera device indices. Aka if we find /dev/video0 and /dev/video2, cap_index == [0,2]
    #     for i in range(0, quantity):
    #         cap_index.append(int(devices[i][-1]))
    #     return cap_index

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


class vStream:
    def __init__(self, src, cam_num, resolution):
        print("Openning camera at link: ", src)
        self.width = resolution[0]
        self.height = resolution[1]
        
        self.capture=cv2.VideoCapture(src)
        # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # success = self.capture.set(cv2.CAP_PROP_FPS, 30.0)
        self.src = cam_num
        self.new_frame_available = False
        self.frame = None
        self.frame_processed = None
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon=True
        self.thread.start()
        


        
    def update(self):
        while True:
            getframe = self.capture.read()
            self.frame = getframe[1]
            # print(self.src, self.capture.get(cv2.CAP_PROP_FPS))
            ## Resizing to set dimensions
            self.frame_resized = cv2.resize(self.frame, (self.width, self.height))
            # frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.new_frame_available = True



    def getFrame(self):
        if self.frame_resized is None:
            return (False, self.frame_resized, self.frame)
        if self.new_frame_available:
            self.new_frame_available = False
            return (True, self.frame_resized, self.frame)
        else:
            return (False, self.frame_resized, self.frame)

