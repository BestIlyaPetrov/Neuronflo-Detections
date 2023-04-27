from threading import Thread
import cv2
import time
import numpy as np



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

w = 640
h = 480
cam1 = vStream(0,w,h)
cam2 = vStream(1,w,h)


while True:
    try:
        myFrame1 = cam1.getFrame()
        myFrame2 = cam2.getFrame()
        # cv2.imshow('Cam1', myFrame1)
        # cv2.imshow('Cam2', myFrame2)
        myFrame3 = np.hstack((myFrame1,myFrame2))
        cv2.imshow('ComboCam', myFrame3)
        cv2.moveWindow('ComboCam',0,0)
    except:
        print('frame unavailable')
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