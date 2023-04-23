import cv2
import torch
import supervision as sv
import argparse
import numpy as np
import operator


# # Load YOLOv5 model
#model = torch.hub.load("ultralytics/yolov5", "yolov5s")
 #Load custom YOLOv5 model from file
model = torch.hub.load('./','custom', path='bestmaskv5.pt', force_reload=True,source='local')


#model_path = "./bestmaskv5.pt"  # Replace with your model file path
#model = torch.load(model_path)

# Open video capture
#cap = cv2.VideoCapture(1)  # Use 0 for default camera or provide video file path
# Open video capture
VIDEO_PATH = './video/horizontal_coming_no_mask.MOV'
path_out='./frame'

ZONE_POLYGON = np.array([
    [500,120],
    [900,120],
    [900,500],
    [500,500]
])

def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv5 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

# Release video capture and destroy any OpenCV windows

def video_to_frames(input_loc, output_loc):

    args = parse_argument()

    cap = cv2.VideoCapture(input_loc)

    zone = sv.PolygonZone(polygon=ZONE_POLYGON, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone,color=sv.Color.blue())
    zone_count=0
    # Start converting the video
    while True:
        # Extract the frame
        ret, frame = cap.read()
        results = model(frame, size=1280)
        detections = sv.Detections.from_yolov5(results)

        # annotate
        box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

        frame = box_annotator.annotate(scene=frame, detections=detections)
        zone.trigger(detections=detections)

        frame = zone_annotator.annotate(scene=frame)

        if(operator.and_(not(zone_count),zone.current_count)):
            if(detections.class_id == 1):
                cv2.imwrite(output_loc + "/mask.jpg" , frame)
            elif(detections.class_id == 0):
                cv2.imwrite(output_loc + "/nomask.jpg" , frame)

        zone_count = zone.current_count
        #cv2.imshow("bestmaskv5.pt",frame)
        if (cv2.waitKey(30)==27):
            break



if __name__=="__main__":

    video_to_frames(VIDEO_PATH, path_out)


