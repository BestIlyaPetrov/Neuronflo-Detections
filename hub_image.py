import torch
import cv2
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', path='bestmaskv5.pt')
model.conf = 0.8
model.iou = 0.45
cam = cv2.VideoCapture(0)

while(True):
    ret, frame = cam.read()
    result = model(frame, size=640)
    dect = result.pandas().xyxy[0]['name']

    cv2.imshow('YOLO', np.squeeze(result.render()))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
