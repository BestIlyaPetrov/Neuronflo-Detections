# from .inference import InferenceSystem
import multiprocessing
import time
import datetime

'''
PURPOSE
This class is used to record up to however long you specify
and be able to access that video whenever called.

HOW TO USE
First you initiate the recorder object specifying which
inference system you want to use. To be able to store frames
you need to make sure that the inference system you are
using has a self.captures attribute. That attribute
must be a list with a length the same number as cameras
available. For example, if there's 3 cameras then 
self.captures will be [[], [], []]. Each element will
be a list that holds a frame from each camera. Calling
the store method will store the most recently captured
frames and inside self.cam_feeds. To be able to get the
frames of a specific camera, make the inference system
have a self.camera_num attribute and make that the
number of the camera index you want. After defining that,
you can call the send method and it will give you the
last few seconds worth of frames for that camera.
'''

class Recorder():
    # def __init__(self, system:InferenceSystem): # Doesn't work due to recursive import - will need to change this
    def __init__(self, system):
        self.system = system
        self.cam_feeds = [[] for _ in range(len(self.system.cams))]
        self.processes = []
    
    def store(self):
        # Will store the most recently captured frames here.  

        # Assuming that the FPS is 15 and we want 5 seconds,
        # the max number of frames we want to collect per 
        # camera will be 75
        record_time = 5 # In seconds
        fps = 15
        max_frames_saved = record_time * fps
        captured_frames = self.system.captures
        num = 0
        for frame in captured_frames:
            # If the number of frames stored for a camera is 
            # the max numbered frames, remove the oldest frame   
            if len(self.cam_feeds[num]) > max_frames_saved-1:
                self.cam_feeds[num].pop(0)
            # Add the newest frame to the list
            self.cam_feeds[num].append(frame)
            # Increment
            num += 1
    
    def send(self):
        # Will start thread to execute code for sending in a video 
        # of the violation 3 seconds before and 2 seconds after
        
        # Will clean up the finished processes 
        self.update()
        # Defines the thread
        process = multiprocessing.Process(target=self.footage, args=())
        # This will be stored in the object's collection of threads
        collection = [datetime.datetime.now(), process]
        self.processes.append(collection)
        # This initializes the process
        process.start()

    
    def footage(self):
        # This will wait 2 seconds after the violation and send 
        # the frames found in that camera's backlog of frames 
        # to the server 
        cam_num = self.system.camera_num
        time.sleep(2)
        footage = self.cam_feeds[cam_num]
        """
        PLACE LOGIC FOR HOW TO SEND VIDEO TO SERVER
        """
        pass

    def update(self):
        # If a thread is younger than 18 seconds, it stays
        self.processes = [collection for collection in self.processes if ((datetime.datetime.now() - collection[0]) < datetime.timedelta(minutes=0.3))] # Extend the time duration if sending to the server takes longer than expected