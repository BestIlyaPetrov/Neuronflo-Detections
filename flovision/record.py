from inference import InferenceSystem

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
    def __init__(self, system:InferenceSystem):
        self.system = system
        self.cam_feeds = [[] for _ in range(len(self.system.cams))]
    
    def store(self):
        # Will store the most recently captured frames here.  

        # Assuming that the FPS is 15 and we want 5 seconds,
        # the max number of frames we want to collect per 
        # camera will be 75
        record_time = 3 # In seconds
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
        # Will send all of the frames recorded for the 
        # last 5 seconds for the current camera.  
        return self.cam_feeds[self.system.camera_num]