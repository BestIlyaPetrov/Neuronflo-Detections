import cv2
import torch
import glob
import requests

MY_DEVICE_ID = 1



# # Load YOLOv5 model
#model = torch.hub.load("ultralytics/yolov5", "yolov5s")
 #Load custom YOLOv5 model from file
model = torch.hub.load('./','custom', path='bestmaskv5.pt', force_reload=True,source='local')


# Find all available video devices
devices = glob.glob('/dev/video*')

# Sort the device names in ascending order
devices.sort()

# Use the first device as the capture index
capture_index = 0

# If there are no devices available, raise an error
if not devices:
    raise ValueError('No video devices found')

# Otherwise, use the lowest available index
else:
    capture_index = int(devices[0][-1])

# Open video capture
cap = cv2.VideoCapture(capture_index)  # Use 0 for default camera or provide video file path

if not cap.isOpened():
    raise ValueError(f"Could not open video device with index {capture_index}")


ret_cnt=0
try:
    while True:
        # Read frame from video
        ret, frame = cap.read()
        #Wait a bit before failing (aka if 30 consequtive frames are dropped, then fail)
        if not ret:
            ret_cnt += 1 
            if ret_cnt >= 30:
                break
            else:
                continue
        ret_cnt = 0 #resets ret_cnt to 0 if ret == True

        # Inference
        results = model(frame)

        # Print results to console
        print(results.pandas().xyxy[0])

         # Save a frame every 10 seconds
        elapsed_time = time.time() - start_time
        
        # Encode the frame data as a JPEG image in memory
        _, buffer = cv2.imencode(".jpg", frame)
        image_data = np.array(buffer).tobytes()

        #Send the image to the server
        sendImageToServer(image_data)
        time.sleep(10)

except KeyboardInterrupt:
    # User interrupted execution
    pass

# Release video capture and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()

def sendImageToServer(image_data):
        # Get authentication token for device from Django
    response = requests.get(f'192.168.0.17:6969/auth/token/?device_id={MY_DEVICE_ID}')
    if response.status_code == 200:
        auth_token = response.text.strip()
    else:
        print('Failed to get authentication token')

    # Set up data to send along with image
    data = {
        'zone_name': '1',
        'crossing_type': 'coming',
    }

    # Set headers including the authentication token
    headers = {
        'Authorization': 'Token {}'.format(auth_token),
    }

    # Send image and data to Django server
    response = requests.post('192.168.0.17:6969/entrance_update/', files={'image': ('image.jpg', image_data)}, data=data, headers=headers)

    # Check response status code
    if response.status_code == 200:
        print('Image uploaded successfully!')
    else:
        print('Failed to upload image')
