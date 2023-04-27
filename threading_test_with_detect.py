        
from threading import Thread
import cv2
import time
import numpy as np
import torch
import json



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
w = 640
h = 480
cam1 = vStream(0,w,h)
cam2 = vStream(1,w,h)


while True:
    try:
        
        # print("CAM1:",cam1.detect())
        # print("CAM2:",cam2.detect())

        myFrame1 = cam1.getFrame()
        myFrame2 = cam2.getFrame()
        # cv2.imshow('Cam1', myFrame1)
        # cv2.imshow('Cam2', myFrame2)
        myFrame3 = np.hstack((myFrame1,myFrame2))
        cv2.imshow('ComboCam', myFrame3)
        cv2.moveWindow('ComboCam',0,0)


        results = model(myFrame3)
        result_dict = results.pandas().xyxy[0].to_dict()
        result_json = json.dumps(result_dict)
        print(result_json)
    except Exception as e:
        print('frame unavailable', e)


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