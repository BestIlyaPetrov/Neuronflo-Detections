from threading import Thread
import cv2
import numpy as np
import json


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




class vStream:
    def __init__(self, src, resolution):
        self.width = resolution[0]
        self.height = resolution[1]
        
        self.capture=cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon=True
        self.thread.start()
    def update(self):
        while True:
            _,self.frame = self.capture.read()
            self.frame2 = cv2.resize(self.frame, (self.width, self.height))

    def getFrame(self):
        return self.frame2

    # def detect(self):
    #     self.results = self.model(self.getFrame())
    #     self.result_dict = results.pandas().xyxy[0].to_dict()
    #     self.result_json = json.dumps(self.result_dict)
    #     return self.result_json