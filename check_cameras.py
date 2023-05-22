from threading import Thread
import cv2
import time
import numpy as np
import traceback


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
                # cv2.moveWindow('ComboCam',0,0)
            except Exception as e:
                print('frame unavailable')
                print("Error is: ", e)
                traceback.print_exc()

            if cv2.waitKey(1) == ord('q'):
                cam1.capture.release()
                cam2.capture.release()
                cv2.destroyAllWindows()
                exit(1)
                break
    except KeyboardInterrupt:
        print()
        print("Keyboard interrupt. Exiting peacefully")
        cam1.capture.release()
        # cam2.capture.release()
        cv2.destroyAllWindows()
        exit(1)


if __name__ == "__main__":
    main()


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