import torch
import cv2
import numpy as np
import pathlib
import asyncio
import websockets
import time
import threading

pastDect = ''

def socket(test):
    # Websocket code
    async def hello(websocket, path, test):
        print("Client connected. Sending messages now.")
        try:
            print("Socket send: ",test)
            await websocket.send(f"Detection: {test}")
            await asyncio.sleep(1)
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected.")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_server = websockets.serve(lambda websocket, path: hello(websocket, path, test=test),"0.0.0.0", 8764)

    try:
        loop.run_until_complete(start_server)
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(asyncio.gather(*asyncio.all_tasks()))
        loop.stop()


def detection():
    # Detection code
    pastDect = ''

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='../trained_models/bestmaskv5.pt')
    model.conf = 0.8
    model.iou = 0.45
    cam = cv2.VideoCapture(0)

    while(True):
        ret, frame = cam.read()
        result = model(frame, size=640)
        dect = result.pandas().xyxy[0]['name']

        try:
            dect = result.pandas().xyxy[0]['name'][0]
            if pastDect != dect:
                pastDect = dect
                socket(pastDect)
                print("PastDect updated: ",pastDect)

        except:
            pass

        cv2.imshow('YOLO', np.squeeze(result.render()))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

detection()