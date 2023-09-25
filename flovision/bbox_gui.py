import cv2
import numpy as np
import json
import os

# Global variable to store points
points = []
polygons = []

def click_event(event, x, y, flags, param):
    global points, polygons
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        img_copy = param.copy()
        if len(points) >= 1 and len(polygons) == 0:
            cv2.polylines(img_copy, [np.array(points)], True, (0, 255, 0), 2)
        elif len(polygons) >= 1:
            for pts in polygons:
                cv2.polylines(img_copy, [np.array(pts)], True, (0, 255, 0), 2) #draw existing polygons
            cv2.polylines(img_copy, [np.array(points)], True, (0, 255, 0), 2) #draw new points
        cv2.imshow("Frame", img_copy)


def save_coordinates(cap_idx):
    global points, polygons
    file_path = f'coordinates{cap_idx}.json'
    # polygons.append(points)
    with open(file_path, 'w') as f:
        json.dump(polygons, f)
    polygons_to_return = polygons.copy()
    polygons = []  # reset the polygons
    points = []  # reset the points
    print("POLYGONS: ", polygons_to_return)
    return polygons_to_return


def create_bounding_boxes(cam):
    global points, polygons
    while True:
        ret, frame = cam.getFrame()
        if ret:
            break

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", click_event, param=frame)

    cv2.imshow("Frame", frame)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Save the current polygon
            if len(points) >= 3:  # If valid polygon
                polygons.append(points.copy())
                points = []
            coordinates = save_coordinates(cam.src)
            cv2.destroyAllWindows()
            break
        elif key == ord('n'):  # start a new polygon
            if len(points) >= 3:  # If valid polygon
                polygons.append(points.copy())
                points = []
            # cv2.imshow("Frame", frame)
        elif key == ord('r'):  # reset the current polygon
            points = []
            cv2.imshow("Frame", frame)
        elif key == ord('q'):  # Quit without saving
            break
    return coordinates


def load_bounding_boxes(cam):
    file_path = f'coordinates{cam.src}.json'
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            polygons = json.load(f)
    else:
        print("Coordinates file does not yet exist for Camera Source ", cam.src)
        print("Please create a new one")
        polygons = create_bounding_boxes(cam)
    return [np.array(polygon) for polygon in polygons]
