from threading import Thread
import cv2
import time
import numpy as np

import tkinter as tk

class PasswordGUI(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.password = "mypassword"  # Replace with your own password
        self.create_widgets()

    def create_widgets(self):
        self.password_label = tk.Label(self, text="Password:")
        self.password_label.pack(side="left")

        self.password_entry = tk.Entry(self, show="*")
        self.password_entry.pack(side="left")

        self.submit_button = tk.Button(self, text="Submit", command=self.submit_password)
        self.submit_button.pack(side="left")

    def submit_password(self):
        if self.password_entry.get() == self.password:
            self.master.destroy()
            main()  # Call your main program function here
        else:
            self.password_entry.delete(0, "end")
            self.password_entry.insert(0, "Incorrect password. Try again.")



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
            ret,self.frame = self.capture.read()
            if ret:
                self.frame2 = cv2.resize(self.frame, (self.width, self.height))

    def getFrame(self):
        return self.frame2

def main():
    w = 320
    h = 240
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



if __name__ == "__main__":
    root = tk.Tk()
    app = PasswordGUI(master=root)
    app.pack()
    app.mainloop()


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