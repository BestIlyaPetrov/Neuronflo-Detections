        
from threading import Thread
import cv2
import time
import numpy as np
import torch, torchvision
import json

import supervision as sv
from utils.general import non_max_suppression

import traceback
from utils.torch_utils import select_device
from models.common import DetectMultiBackend


# VIDEO RESOLUTION
video_res = [640, 480]

# NMS paparms
conf_thres = 0.5
iou_thres = 0.7
agnostic_nms= True
classes = None ## all classes are detected

#Image Annotation Params
border_thickness = 15

#Region for goggles
w1 = 30 
h1 = 100
x1 = 20
y1 = 50

#Region for shoes
w2 = 30 
h2 = 100
x2 = 20
y2 = 50


CENTER_COORDINATES = [] #Center of the detection region as percentage of FRAME_SIZE
WIDTH = []#% of the screen 
HEIGHT = [] #% of the screen 

CENTER_COORDINATES.append((x1 /2,y1)) #Center of the detection region as percentage of FRAME_SIZE
WIDTH.append(w1 /2) #% of the 1st screen 
HEIGHT.append(h1) #% of the 1st screen 

CENTER_COORDINATES.append((50+( x2 /2),y2)) #Center of the detection region as percentage of FRAME_SIZE
WIDTH.append(w2 /2) #% of the 2nd screen 
HEIGHT.append(h2) #% of the 2nd screen 


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


# device = select_device('0')
# model = DetectMultiBackend('bestmaskv5.pt', device=device)

#Initialize the cameras 
cam1 = vStream(0,video_res)
cam2 = vStream(1,video_res)

#Load the model
model = torch.hub.load('./','custom', path='bestmaskv5.pt', force_reload=True,source='local', device='0')

frame_size=(2*w,h) #since we are horizonatlly stacking the two images
# Calculate detection region
zone_polygons = []
zone_polygons.append(region_dimensions(frame_size, CENTER_COORDINATES[0], WIDTH[0], HEIGHT[0]))
zone_polygons.append(region_dimensions(frame_size, CENTER_COORDINATES[1], WIDTH[1], HEIGHT[1]))

colors = sv.ColorPalette.default()
zones = [
    sv.PolygonZone(
        polygon=polygon, 
        frame_resolution_wh=frame_size
    )
    for polygon
    in zone_polygons
]
zone_annotators = [
    sv.PolygonZoneAnnotator(
        zone=zone, 
        color=colors.by_idx(index+2), 
        thickness=4,
        text_thickness=2,
        text_scale=2
    )
    for index, zone
    in enumerate(zones)
]
# box_annotators = [
#     sv.BoxAnnotator(
#         color=colors.by_idx(index), 
#         thickness=4, 
#         text_thickness=4, 
#         text_scale=2
#         )
#     for index
#     in range(len(zone_polygons))
# ]

# zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=frame_size)
# zone_annotator = sv.PolygonZoneAnnotator(zone=zone,color=sv.Color.blue()) 
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

zone_count=0

while True:
    try:
        
        myFrame1 = cam1.getFrame()
        myFrame2 = cam2.getFrame()
        # cv2.imshow('Cam1', myFrame1)
        # cv2.imshow('Cam2', myFrame2)
        frame = np.hstack((myFrame1,myFrame2))
        

        # Run Inference
        results = model(frame)

        #load results into supervision
        detections = sv.Detections.from_yolov5(results)
        #Apply NMS to remove double detections
        detections = detections.with_nms(threshold=0.5, class_agnostic=True) #apply NMS to detections

         #Get data ready
        compliant=False
        if(detections.class_id.any() == 0):
            compliant=False #no_mask
        elif(detections.class_id.all() == 1):
            compliant=True #mask

        bordered_frame = draw_border(myFrame2, compliant, border_thickness)
        #Annotate
        for zone, zone_annotator in zip(zones, zone_annotators):
            zone.trigger(detections=detections)
            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = zone_annotator.annotate(scene=frame)
        # frame = box_annotator.annotate(scene=frame, detections=detections)

        # zone.trigger(detections=detections)

        # frame = zone_annotator.annotate(scene=frame)
        #Display frame
        cv2.imshow('ComboCam', bordered_frame)

        # result_dict = results.pandas().xyxy[0].to_dict()
        # result_json = json.dumps(result_dict)
        # print(result_json)
    except Exception as e:
        print('frame unavailable', e)
        traceback.print_exc()


    if cv2.waitKey(1) == ord('q'):
        cam1.capture.release()
        cam2.capture.release()
        cv2.destroyAllWindows()
        exit(1)
        break


