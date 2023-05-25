from threading import Thread
import cv2
import time
import numpy as np
import traceback
import glob
import argparse



class vStream:
    def __init__(self, src, width, height):
        self.width = width
        self.height = height
        self.capture=cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.is_recording = False
        self.out = None
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon=True
        self.thread.start()
        self.src = src

    def update(self):
        while True:
            _, self.frame = self.capture.read()
            try:
                if self.is_recording:
                    self.out.write(self.frame)
            except Exception as e:
                print("Error writing frame:", e)

            self.frame2 = cv2.resize(self.frame, (self.width, self.height))

    def start_recording(self, output_file):
        self.is_recording = True
        self.out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (self.width, self.height))

    def stop_recording(self):
        self.is_recording = False
        if self.out is not None:
            self.out.release()

    def getFrame(self):
        return self.frame2

def main(
    display_video = False
    ):

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

        choice = input(f"Enter the number of the video device you want to use: ")
        choice = int(choice)

        if choice < 1 or choice > num_devices:
            print("Invalid choice.")
            return

        # Subtracting 1 to match list indices
        selected_device = devices[choice - 1][-1]

        cam = vStream(int(selected_device), w, h)

        is_recording = False
        output_file = None

        while True:
            try:
                myFrame = cam.getFrame()
                if display_video:
                    cv2.imshow('Camera View', myFrame)
            except Exception as e:
                print('Frame unavailable')
                print("Error:", e)
                traceback.print_exc()

            key = cv2.waitKey(1)

            if key == ord('r') and not is_recording:
                is_recording = True
                timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                output_file = f"recorded_videos/recorded_video_{timestamp}.mp4"
                cam.start_recording(output_file)
                print("Started recording...")

            elif key == ord('s') and is_recording:
                is_recording = False
                cam.stop_recording()
                print("Stopped recording and saved the video.")

            if key == ord('q'):
                break

        cam.capture.release()
        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print()
        print("Keyboard interrupt. Exiting peacefully")
        cam.capture.release()
        cv2.destroyAllWindows()
        exit(1)

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--display-video', action='store_true', help='show video feed')

    options = parser.parse_args()
    return options

if __name__ == "__main__":
    opt = parse_options()
    main(**vars(opt))