import face_recognition
import cv2
import numpy as np
from utils import *

"""
This file helps guide how to use the face_recog
class and give an example below
"""

class face_recog:
    """
    This is a helper class for the facial recognition.
    This class will be able to store face encodings and names,
    VideoCapture objects, the resolution of the camera, and
    the directory of the database. With these stored values,
    this class will be able to give you frames,

    capture: Takes in the source for a VideoCapture object
    directory: Takes in a string that leads to the database directory
    """
    def __init__(self, capture, directory:str):
        self.cam = cv2.VideoCapture(capture)
        self.width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.directory = directory
        self.known_face_encodings = []
        self.known_face_names = []
        self.initialize()

    def initialize(self):
        people_collections = list_files_recursively(self.directory)
        for person_collection in people_collections:
            for person, image_paths in person_collection.items():
                mixture_encoding = face_mixture_encoding(image_paths)
                self.known_face_encodings.append(mixture_encoding)
                self.known_face_names.append(person)
    
    def release(self):
        self.cam.release()

    def FindFaceEncodings(self):
        ret, frame = self.cam.read()
        if ret:
            # Convert the image from BGR color (which OpenCV uses) to
            # RGB color (which face_recognition uses)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            return None, None, None
        
        # Find all the faces and face enqcodings in the frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        return frame, face_encodings, face_locations
    
    def show(self, frame, face_encodings, face_locations):
        close_window = False

        face_info = zip(face_locations, face_encodings)
        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in face_info:
            # See if the face is a match for the known face(s)
            norm_top = top/self.height
            norm_bottom = bottom/self.height
            norm_left = left/self.width
            norm_right = right/self.width

            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding) # [False, True, False]
            # print(matches)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding) #[0.5 0.3 0.7]
            # print(face_distances)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                # PLACE RESTRICTIONS RIGHT HERE!!!!
                name = self.known_face_names[best_match_index]

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Waits for when someone hits the 'q' key to exit the program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                close_window = True
                break
        return frame, close_window
    
if __name__ == "__main__":
    # How to initiate an instance of face_recog
    face_classifier = face_recog(0, "../database/")

    # A while loop for face recognition
    while True:
        frame, face_encodings, face_locations = face_classifier.FindFaceEncodings()
        if frame is None:
            continue
        
        frame, close_window = face_classifier.show(frame, face_encodings, face_locations)
        if close_window:
            break

        cv2.imshow("face_recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    face_classifier.release()
    del face_classifier
    cv2.destroyAllWindows()