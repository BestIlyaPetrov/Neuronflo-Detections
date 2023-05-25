from threading import Thread
import cv2
import time
import numpy as np
import traceback
import glob


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
        self.src = src
    def update(self):
        while True:
            # print(self.src, self.capture.get(cv2.CAP_PROP_FPS))
            _,self.frame = self.capture.read()
            try:
                print(self.src, self.capture.get(cv2.CAP_PROP_FPS))
            except:
                print("Couldn't read fps on src #",src)
            self.frame2 = cv2.resize(self.frame, (self.width, self.height))

    def getFrame(self):
        return self.frame2

def main():
    try:
        w = 640
        h = 480

        # Find all available video devices
        devices = glob.glob('/dev/video*')
        num_devices = len(devices)

        if num_devices == 0:
            print("No video devices found.")
            return

        print("Available video devices:")
        for i, device in enumerate(devices):
            print(f"{i+1}. {device}")

        choice = input("Enter the number of the video device you want to use: ")
        choice = int(choice)

        if choice < 1 or choice > num_devices:
            print("Invalid choice.")
            return

        # Subtracting 1 to match list indices
        selected_device = devices[choice - 1][-1]

        cam = vStream(selected_device, w, h)

        while True:
            try:
                myFrame = cam.getFrame()
                cv2.imshow('Camera View', myFrame)
            except Exception as e:
                print('Frame unavailable')
                print("Error:", e)
                traceback.print_exc()

            if cv2.waitKey(1) == ord('q'):
                cam.capture.release()
                cv2.destroyAllWindows()
                break
    except KeyboardInterrupt:
        print()
        print("Keyboard interrupt. Exiting peacefully")
        cam.capture.release()
        cv2.destroyAllWindows()
        exit(1)


if __name__ == "__main__":
    main()
