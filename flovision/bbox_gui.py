import cv2
import numpy as np
import json
import os

# Global variable to store points
points = []

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) >= 3:
            # Draw polygon
            img_copy = param.copy()
            cv2.polylines(img_copy, [np.array(points)], True, (0, 255, 0), 2)
            cv2.imshow("Frame", img_copy)
            
        # else:
        #     # Draw the points
        #     cv2.circle(param, (x,y), 5, (0,0,255), -1)
        # cv2.imshow("Frame", param)

def save_coordinates(cap_idx):
    global points
    file_path = f'coordinates{cap_idx}.json'
    with open(file_path, 'w') as f:
        json.dump(points, f)
        pts_to_return  = points.copy()
        points = [] # reset the points
    return np.array(pts_to_return)

def create_bounding_boxes(cam):
    global points
    # cap = cv2.VideoCapture(1)
    # ret, frame = cap.read()
    while True:
        ret, frame = cam.getFrame()
        if ret:
            break

    # if ret:
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", click_event, param=frame)

    cv2.imshow("Frame", frame)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            coordinates = save_coordinates(cam.src)
            cv2.destroyAllWindows()
            break
        elif key == ord('r'):
            points = []
            cv2.imshow("Frame", frame)
        elif key == ord('q'):
            break
    return coordinates
    # cap.release()
    # cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
def load_bounding_boxes(cam):

    file_path = f'coordinates{cam.src}.json'
    if os.path.isfile(file_path):
        # Load the coordinates from the JSON file
        with open(file_path, 'r') as f:
            coordinates = json.load(f)
    else:
        print("Coordinates file does not yet exist for Camera Source ", cam.src)
        print("Please create a new one")
        coordinates = create_bounding_boxes(cam)

    # Convert the coordinates to a NumPy array
    return np.array(coordinates)
