import os
import traceback

import socket
from zeroconf import ServiceBrowser, Zeroconf

import cv2
# import face_recognition # Causes problems on Windows

class IPListener:
    def __init__(self, target_service_name):
        self.target_service_name = target_service_name
        self.found_ip = None

    def remove_service(self, zeroconf, type, name):
        pass

    def add_service(self, zeroconf, type, name):
        info = zeroconf.get_service_info(type, name)
        if info:
            addresses = [socket.inet_ntoa(addr) for addr in info.addresses]
            print(f"Discovered service: {name}, IP addresses: {', '.join(addresses)}")
            
        if info and name == self.target_service_name:
            address = socket.inet_ntoa(info.addresses[0])
            self.found_ip = address
            print(f"Found IP address of {name}: {address}")

    def update_service(self, zeroconf, type, name):
        pass


def findLocalServer(target_service_name = "neuronflo-server._workstation._tcp.local."):
    #Find the IP of the windows server
    listener = IPListener(target_service_name)
    zeroconf = Zeroconf()

    try:
        print("Looking for server IP...")
        browser = ServiceBrowser(zeroconf, "_workstation._tcp.local.", listener)
        while True:
            if listener.found_ip is not None:
                break
    except KeyboardInterrupt:
        pass
    finally:
        zeroconf.close()

    # Continue with more lines of code after the IP address has been found
    print(f"IP address found: {listener.found_ip}, continuing with the rest of the script...")
    return listener.found_ip



#returns the highest image index in a directory
def get_highest_index(folder_path):
    max_index = -1

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            try:
                index = int(filename.split(']')[0].split('[')[1])
                if index > max_index:
                    max_index = index
            except Exception as e:#(ValueError, IndexError):
                print("Couldn't parse the image index in: ", folder_path)
                print(e)
                traceback.print_exc()
                pass

    return max_index

def list_files_recursively(directory):
    # Creates a list of dictionaries that will hold the pathways for each
    # person's picture to be used for face encoding
    """
    Directory
        Person_1_Folder
            Picture_of_Person_1
        Person_2_Folder
            Picture_of_Person_2
        Person_3_Folder
            Picture_of_Person_3
    """
    
    files_and_dirs = os.listdir(directory)
    persons_dictionary = []
    for item in files_and_dirs:
        item_path = os.path.join(directory, item)

        # If the item is a directory, recursively call the function to list its files
        if os.path.isdir(item_path):
            files_and_dirs2 = os.listdir(item_path)
            image_paths = []
            
            for item2 in files_and_dirs2:
                item_path2 = os.path.join(item_path, item2)
                image_paths.append(item_path2)
            
            person_dictionary = {item: image_paths}
            persons_dictionary.append(person_dictionary)
    
    return persons_dictionary

"""
directory_path = './database/'
list_files_recursively(directory_path)

Will return this:
[
 {
  'Biden':  ['./database/Biden/biden.jpg']
 }, 
 {
  'Sialoi': ['./database/Sialoi/Sialoi8.jpg', 
             './database/Sialoi/Sialoi3.jpg', 
             './database/Sialoi/Sialoi2.jpg', 
             './database/Sialoi/Sialoi1.jpg', 
             './database/Sialoi/Sialoi5.jpg', 
             './database/Sialoi/Sialoi4.jpg', 
             './database/Sialoi/Sialoi6.jpg', 
             './database/Sialoi/Sialoi7.jpg']
 }, 
 {
  'Obama':  ['./database/Obama/obama.jpg']
 }
]

This means that the list_files_recursively function will output
keys that will be the person's name and the the pictures
associated to them for the face encoding.
"""

def face_mixture_encoding(image_paths:list):
    """
    A function to get an accurate encoding for inputting
    multiple images for a single person to be used later
    during face detections.
    
    Needs the list input from the list_files_recursively()
    function.
    """
    # Load the images of the person you want to create an encoding for
    if len(image_paths) > 1:
        images = [face_recognition.load_image_file(image_path) for image_path in image_paths]
        
        # Loop through the face_encodings list and check if it has elements before accessing the first element.
        face_encodings = [face_recognition.face_encodings(image) for image in images]
        face_encodings = [face_encoding[0] if len(face_encoding) > 0 else 0 for face_encoding in face_encodings]

    else:
        images = face_recognition.load_image_file(image_paths)
        
        # Compute the face encodings for each image
        face_encodings = face_recognition.face_encodings(images)[0]

    # Calculate the average face encoding
    average_face_encoding = sum(face_encodings) / len(face_encodings)
    
    # Now you have the average_face_encoding, which represents the person's face
    # You can use this encoding for face recognition later
    return average_face_encoding

def capture_and_save_image():
    """
    A function to capture new pictures for the database.
    No input arguments needed.
    """
    # Open the default camera (index 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera was opened successfully
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    # Create a window to display the camera feed
    cv2.namedWindow("Camera Feed")

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        # Check if the frame was captured successfully
        if not ret:
            print("Error: Failed to capture a frame from the camera.")
            break

        # Display the frame in the window
        cv2.imshow("Camera Feed", frame)

        # Check for the Spacebar key press
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # ASCII value of Spacebar
            # Generate a new file name with the current timestamp
            import time
            timestamp = int(time.time())
            image_filename = f"capturedimage{timestamp}.jpg"

            # Save the image with the new file name
            cv2.imwrite(image_filename, frame)

            print(f"Image captured and saved as '{image_filename}'.")

        # Check for the 'q' key press to quit the loop
        if key == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

