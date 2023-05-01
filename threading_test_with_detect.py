        
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

# NMS paparms
conf_thres =0.5
iou_thres = 0.7
agnostic_nms= True
classes = None ## all classes are detected

CENTER_COORDINATES = (10,50) #Center of the detection region as percentage of FRAME_SIZE
WIDTH = 40 #% of the screen 
HEIGHT = 100 #% of the screen 




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
    def __init__(self, src, width, height):
        self.width = width
        self.height = height
        
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

model = torch.hub.load('./','custom', path='bestmaskv5.pt', force_reload=True,source='local', device='0')
# device = select_device('0')
# model = DetectMultiBackend('bestmaskv5.pt', device=device)
w = 640
h = 480
cam1 = vStream(0,w,h)
cam2 = vStream(1,w,h)

frame_size=(2*w,h) #since we are horizonatlly stacking the two images
# Calculate detection region
zone_polygon = region_dimensions(frame_size, CENTER_COORDINATES, WIDTH, HEIGHT)

zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=frame_size)
zone_annotator = sv.PolygonZoneAnnotator(zone=zone,color=sv.Color.blue())
zone_count=0

while True:
    try:
        
        # print("CAM1:",cam1.detect())
        # print("CAM2:",cam2.detect())

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


        # annotate
        box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

        frame = box_annotator.annotate(scene=frame, detections=detections)

        zone.trigger(detections=detections)

        frame = zone_annotator.annotate(scene=frame)
        #Display frame
        cv2.imshow('ComboCam', frame)

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


# while True:
#     cap = cv2.VideoCapture(0)
    
#     try:
#         ret,frame = cap.read()
#         cv2.imshow('Cam2', frame)
#     except:
#         print('bitch')

#     if cv2.waitKey(1) == ord('q'):
#         cap.release()
#         cv2.destroyAllWindows()
#         exit(1)
#         break