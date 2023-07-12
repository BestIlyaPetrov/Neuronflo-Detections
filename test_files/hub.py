import torch
import cv2
import numpy as np
import pathlib
import asyncio
import websockets
import time
import threading

pastDect = ''
pastDect


def socket():
    global pastDect

    # Websocket code
    async def hello(websocket, path):
        print("Client connected. Sending messages now.")
        try:
            print("Socket sesnd: ",pastDect)
            await websocket.send(f"Detection: {pastDect}")
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected.")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_server = websockets.serve(lambda websocket, path: hello(websocket, path),"0.0.0.0", 8764)

    try:
        loop.run_until_complete(start_server)
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(asyncio.gather(*asyncio.all_tasks()))
        loop.stop()


def detection():
    global pastDect

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


socketThread = threading.Thread(target=socket)
detectionThread = threading.Thread(target=detection)


detectionThread.start()
# starting thread 1
socketThread.start()
# starting thread 2
