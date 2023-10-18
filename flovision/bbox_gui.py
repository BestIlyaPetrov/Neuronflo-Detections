import cv2
import numpy as np
import json
import os
import glob



# Global variable to store points
points = []

# Preset colors and an iterator to cycle through them
preset_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
color_iter = iter(preset_colors)

def get_next_color():
    global color_iter
    try:
        return next(color_iter)
    except StopIteration:
        # If we've used all preset colors, restart the iterator
        color_iter = iter(preset_colors)
        return next(color_iter)
    
def overlay_commands_on_frame(frame, src):
    '''Overlay commands with a single semi-transparent background on the frame.'''
    font = cv2.FONT_HERSHEY_DUPLEX 
    commands = [f"CAMERA #{src}", "r - reset polygon", "s - save polygon", "q - finish frame"]
    
    # Determine the total space needed for all three text labels
    total_width = 0
    total_height = 0
    for command in commands:
        (w, h), _ = cv2.getTextSize(command, font, 0.5, 2)
        total_width = max(total_width, w)
        total_height += h

    # Define margin for the text background
    x_start = 10
    y_start = 10
    box_margin = 5
    space_between_text = 20
    y_offset = 20

    # Calculate background position based on the bottom-most text placement
    top_left = (x_start - box_margin, y_start - box_margin)
    bottom_right = (x_start + total_width + box_margin, total_height + len(commands) * space_between_text + box_margin)
    
    # Draw semi-transparent rectangle for background
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)  # Blend images

    # Draw the command text based on original placement
    
    for command in commands:
        cv2.putText(frame, command, (x_start, y_start + y_offset), font, 0.5, (255, 255, 255), 1)
        y_offset += 20
        
    return frame





def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        img_copy = param.copy()

        # Always use green for the in-progress polygon
        in_progress_color = (0, 255, 0)

        if len(points) == 1:
            cv2.circle(img_copy, points[0], 5, in_progress_color, -1)
        elif len(points) == 2:
            cv2.line(img_copy, points[0], points[1], in_progress_color, 2)
        else:  # 3 or more points
            cv2.polylines(img_copy, [np.array(points)], False, in_progress_color, 2)  # False ensures the polygon isn't closed yet
            if len(points) >= 3:
                cv2.polylines(img_copy, [np.array(points)], True, in_progress_color, 2)  # True will close the polygon

        cv2.imshow("Frame", img_copy)



def save_coordinates(cap_idx, zone_idx, img):
    global points
    file_path = f'coordinates_{cap_idx}_{zone_idx}.json'
    with open(file_path, 'w') as f:
        json.dump(points, f)
    current_color = get_next_color()  # Get the next color from the preset list
    cv2.polylines(img, [np.array(points)], True, current_color, 2)
    pts_to_return = points.copy()
    points = []  # reset the points
    return np.array(pts_to_return), img




def create_bounding_boxes(cam):
    global points
    coordinates_set = []
    last_saved_coords = None  # To track the last saved set of coordinates

    while True:
        ret, frame = cam.getFrame()
        if ret:
            break

    frame_with_commands = overlay_commands_on_frame(frame,cam.src)  # Apply commands to the frame
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", click_event, param=frame_with_commands)
    cv2.imshow("Frame", frame_with_commands)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if last_saved_coords != points and points:  # Check if the new points are different from the last saved ones
                print(f"Saved the following coordinates", points)
                coords, frame_with_commands = save_coordinates(cam.src, len(coordinates_set), frame_with_commands)
                last_saved_coords = points.copy()  # Update the last saved coordinates
                coordinates_set.append(coords)
                cv2.imshow("Frame", frame_with_commands)
        elif key == ord('r'):
            points = []
            last_saved_coords = None  # Reset the last saved coordinates on reset
            frame_with_commands = overlay_commands_on_frame(frame.copy(),cam.src)
            cv2.imshow("Frame", frame_with_commands)
        elif key == ord('q'):
            if last_saved_coords != points and points:  # Check if the new points are different from the last saved ones
                print(f"Saved the following coordinates", points)
                coords, frame_with_commands = save_coordinates(cam.src, len(coordinates_set), frame_with_commands)
                last_saved_coords = points.copy()  # Update the last saved coordinates
                coordinates_set.append(coords)
                cv2.imshow("Frame", frame_with_commands)
            break
    
    cv2.destroyAllWindows()
    return coordinates_set




    # cap.release()
    # cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
def load_bounding_boxes(cam):
    paths = glob.glob(f'coordinates_{cam.src}_*.json')
    num_zones = len(paths)
    
    if num_zones == 0:
        print("No bounding boxes found for Camera Source ", cam.src)
        print("Please create a new one")
        coordinates_set = create_bounding_boxes(cam)
        return coordinates_set
    
    coordinates_set = []
    for path in paths:
        with open(path, 'r') as f:
            coordinates_set.append(np.array(json.load(f)))

    return coordinates_set