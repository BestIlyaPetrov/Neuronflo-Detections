import cv2
import torch

# # Load YOLOv5 model
#model = torch.hub.load("ultralytics/yolov5", "yolov5s")
 #Load custom YOLOv5 model from file
model = torch.hub.load('./','custom', path='bestmaskv5.pt', force_reload=True,source='local')


#model_path = "./bestmaskv5.pt"  # Replace with your model file path
#model = torch.load(model_path)

# Open video capture
cap = cv2.VideoCapture(1)  # Use 0 for default camera or provide video file path

try:
    while True:
        # Read frame from video
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        results = model(frame)

        # Print results to console
        print(results.pandas().xyxy[0])

except KeyboardInterrupt:
    # User interrupted execution
    pass

# Release video capture and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()
