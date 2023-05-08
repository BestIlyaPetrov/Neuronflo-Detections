from threading import Thread
import cv2
import time
import numpy as np
import torch, torchvision
import json

import supervision as sv
import traceback

from .video import draw_border, region_dimensions, vStream
from .comms import sendImageToServer


class InferenceSystem:
    def __init__(self, model_name, video_res, CENTER_COORDINATES, WIDTH, HEIGHT, border_thickness):
        self.initialize(model_name, video_res, CENTER_COORDINATES, WIDTH, HEIGHT, border_thickness)

    def initialize(self, model_name, video_res, CENTER_COORDINATES, WIDTH, HEIGHT, border_thickness):
        # Initialize the cameras
        self.cam1 = vStream(0, video_res)
        self.cam2 = vStream(1, video_res)

        # Load the model
        self.model = torch.hub.load('./', 'custom', path='bestmaskv5.pt', force_reload=True, source='local', device='0')

        # Set frame params
        self.frame_size = (video_res[0] * 2, video_res[1])  # since we are horizontally stacking the two images
        self.border_thickness = border_thickness
        # Calculate detection region
        zone_polygons = []
        for i in range(len(CENTER_COORDINATES)):
            zone_polygons.append(region_dimensions(self.frame_size, CENTER_COORDINATES[i], WIDTH[i], HEIGHT[i]))

        # set the zones
        colors = sv.ColorPalette.default()
        self.zones = [
            sv.PolygonZone(
                polygon=polygon,
                frame_resolution_wh=self.frame_size
            )
            for polygon
            in zone_polygons
        ]
        self.zone_annotators = [
            sv.PolygonZoneAnnotator(
                zone=zone,
                color=colors.by_idx(index + 2),
                thickness=4,
                text_thickness=2,
                text_scale=2
            )
            for index, zone
            in enumerate(self.zones)
        ]

        self.box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

    def run(self):
        zone_count = 0

        while True:
            try:
                myFrame1 = self.cam1.getFrame()
                myFrame2 = self.cam2.getFrame()
                frame = np.hstack((myFrame1, myFrame2))

                # Run Inference
                results = self.model(frame)

                # load results into supervision
                detections = sv.Detections.from_yolov5(results)
                # Apply NMS to remove double detections
                detections = detections.with_nms(threshold=0.5, class_agnostic=True)  # apply NMS to detections

                # Annotate
                for zone, zone_annotator in zip(self.zones, self.zone_annotators):
                    zone.trigger(detections=detections)
                    frame = self.box_annotator.annotate(scene=frame, detections=detections)
                    frame = zone_annotator.annotate(scene=frame)


                if self.zones[0].current_count > zone_count:

                    # Get data ready
                    compliant = False
                    if (detections.class_id.any() == 0):
                        compliant = False  # no_mask
                    elif (detections.class_id.all() == 1):
                        compliant = True  # mask
                    bordered_frame = draw_border(myFrame2, compliant, self.border_thickness)

                    data = {
                            'zone_name': '1',
                            'crossing_type': 'coming',
                            'compliant' : str(compliant)
                        }

                    # JSON RAW DATA
                    # Convert pandas DataFrame to a Python dictionary
                    result_dict = results.pandas().xyxy[0].to_dict()
                    # Convert dictionary to JSON string
                    result_json = json.dumps(result_dict)
                    # Print the JSON string
                    print()
                    print("RESULTS: ", results)
                    print("RESULTS JSON: ", result_json)
                    print()

                    #convert image to be ready to be sent
                    success, encoded_image = cv2.imencode('.jpg', bordered_frame)    
                    if success:
                        # Convert the encoded image to a byte array
                        image_bytes = bytearray(encoded_image)
                        # You can now use image_data like you did with f.read() 

                        # Send the image to the server
                        sendImageToServer(image_bytes, data)

                        print()
                        print("########### DETECTION MADE #############")
                        print(result_json)
                        print("########### END OF DETECTION #############")
                        print()
                    else:
                        raise ValueError("Could not encode the frame as a JPEG image")



                



                zone_count = self.zones[0].current_count
                # Display frame
                # cv2.imshow('ComboCam', bordered_frame)

            except Exception as e:
                print('frame unavailable', e)
                traceback.print_exc()

            if cv2.waitKey(1) == ord('q'):
                self.cam1.capture.release()
                self.cam2.capture.release()
                cv2.destroyAllWindows()
                exit(1)
               
