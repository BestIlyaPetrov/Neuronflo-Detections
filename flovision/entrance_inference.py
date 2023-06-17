from threading import Thread
import cv2
import time
import numpy as np
import torch, torchvision
import json

import supervision as sv
import traceback
import collections
from pathlib import Path

from .bbox_gui import create_bounding_boxes, load_bounding_boxes
from .video import draw_border, region_dimensions, vStream, least_blurry_image_indx, get_device_indices
from .comms import sendImageToServer
from .utils import get_highest_index, findLocalServer
from .peripherals import ping_alarm
from zeroconf import ServiceBrowser, Zeroconf



class EntranceInferenceSystem:
    def __init__(self, model_name, video_res, border_thickness, display, save, bboxes):
        self.initialize(model_name, video_res, border_thickness, display, save, bboxes)

    def initialize(self, model_name, video_res, border_thickness, display, save, bboxes):


        self.server_IP = findLocalServer()
        cap_index = get_device_indices(quantity = 2)

        # Initialize the cameras
        self.cams = []
        self.cams.append(vStream(cap_index[0], video_res))
        self.cams.append(vStream(cap_index[1], video_res))

        # Define the detection regions
        zone_polygons = []
        if bboxes:
            coordinates = create_bounding_boxes(self.cams[0])
            zone_polygons.append(coordinates)

            coordinates = create_bounding_boxes(self.cams[1])
            coordinates[:, 0] += video_res[0] # add 640 to each x coordinates, bc the frames are horizonatally stacked
            zone_polygons.append(coordinates)
        else:
            coordinates = load_bounding_boxes(self.cams[0])
            zone_polygons.append(coordinates)

            coordinates = load_bounding_boxes(self.cams[1])
            coordinates[:, 0] += video_res[0] # add 640 to each x coordinates, bc the frames are horizontally stacked
            zone_polygons.append(coordinates)
            


        # Load the model
        self.model = torch.hub.load('./', 'custom', path=model_name, force_reload=True, source='local', device='0')

        # Set frame params
        self.frame_size = (video_res[0] * 2, video_res[1])  # since we are horizontally stacking the two images
        self.border_thickness = border_thickness
        self.display = display
        self.save = save
 

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

        self.box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)

    def stop(self):
        print("Stopping detections and releasing cameras")
        for i in range(0, len(self.cams)):
            self.cams[i].capture.release()
        cv2.destroyAllWindows()
        exit(1)


    def save_frames(self, frame_arr):
        try:
            # Path of the directory
            dir_path = Path.cwd().parent / 'saved_frames'
            # Create the directories if it doesn't exist
            goggle_path = dir_path / 'goggles'
            shoe_path = dir_path / 'shoes'
            goggle_path.mkdir(parents=True, exist_ok=True)
            shoe_path.mkdir(parents=True, exist_ok=True)
            count = get_highest_index(goggle_path) + 1 
            for img in frame_arr[0]:
                cv2.imwrite(str(goggle_path / f"goggles_img_{count:04d}.jpg"), img)
                count += 1 
            count = get_highest_index(shoe_path) + 1
            for img in frame_arr[1]:
                cv2.imwrite(str(shoe_path / f"shoes_img_{count:04d}.jpg"), img)
                count += 1
            

        except Exception as e:
            print("Couldn't save the frames.")
            print(e)
            traceback.print_exc()

    def run(self, iou_thres, agnostic_nms):
        print("Inference successfully launched")
        zone_count = 0
        n = 3 # num_consecutive_frames that we want to window (to reduce jitter)
        detections_array = []
        frame1_array = []
        frame2_array = []
        cnt = 0
        detection_trigger_flag = False
        # fps_list = []
        while True:
            try:
                # start the timer
                # start_time = time.time()

                ## Make the slowest cam be the bottleneck here
                new_frame1, myFrame1 = self.cams[0].getFrame()
                if new_frame1 == False:
                    continue
                new_frame2, myFrame2 = self.cams[1].getFrame()
                if new_frame2 == False:
                    continue

                frame = np.hstack((myFrame1, myFrame2))

                # Run Inference
                results = self.model(frame)

                # load results into supervision
                detections = sv.Detections.from_yolov5(results)
                # Apply NMS to remove double detections
                detections = detections.with_nms(threshold=iou_thres,  class_agnostic=agnostic_nms)  # apply NMS to detections

                

                # Annotate
                mask = []
                for zone, zone_annotator in zip(self.zones, self.zone_annotators):
                    mask.append(zone.trigger(detections=detections))
                    frame = self.box_annotator.annotate(scene=frame, detections=detections)
                    frame = zone_annotator.annotate(scene=frame)

                # Split into two sets of detections by bounding box
                goggle_classes_set = [0, 1] #goggles, no_goggles
                shoe_classes_set = [2, 3] #safe_shoes, not_safe_shoes
                goggle_det = detections[mask[0] & np.isin(detections.class_id, goggle_classes_set)] # mask[0] is the left bounding box
                shoe_det = detections[mask[1] & np.isin(detections.class_id, shoe_classes_set)] # mask[1] is the right bounding box


                # TRIGGER EVENT
                if self.zones[0].current_count > zone_count and detection_trigger_flag==False:
                    detection_trigger_flag = True

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


                # Once the first face is detected we collect the next n frames, take the most occured detection and use the least blurry image
                if detection_trigger_flag == True:
                    cnt +=1
                    detections_array.append((goggle_det, shoe_det))
                    frame1_array.append(myFrame1)
                    frame2_array.append(myFrame2)
                    # print("Goggle detections: ", goggle_det)
                    # print("Shoe detections: ", shoe_det)
                    if cnt >= n: #n frames after first detection trigger
                        cnt = 0
                        detection_trigger_flag=False
                        goggle_detections = []
                        shoe_detections = []
                        # print(detections_array)
                        for goggle_det, shoe_det in detections_array:
                            if hasattr(goggle_det, 'class_id') and len(goggle_det.class_id > 0):
                                [goggle_detections.append(int(ids)) for ids in goggle_det.class_id]
                                # print("len:",len(goggle_det.class_id))
                                # print("goggle detections: ",goggle_detections)
                            if hasattr(shoe_det, 'class_id') and len(shoe_det.class_id > 0):
                                [shoe_detections.append(int(ids)) for ids in shoe_det.class_id]
                                # print("shoe detections: ",shoe_det)
                           

                        most_common_goggle_detection = 0 #no goggles
                        most_common_shoe_detection = 2  #wrong_shoes
                        if len(goggle_detections):
                            most_common_goggle_detection = collections.Counter(goggle_detections).most_common(1)[0][0]
                            print("Most common goggle detection: ", most_common_goggle_detection)
                        else:
                            print("No goggle detections made")

                        if len(shoe_detections):
                            most_common_shoe_detection = collections.Counter(shoe_detections).most_common(1)[0][0]
                            print("Most common shoe detection: ", most_common_shoe_detection)
                        else:
                            print("No shoe detections made")

                        #Pick least blurry image
                        myFrame1 = frame1_array[least_blurry_image_indx(frame1_array)]
                        myFrame2 = frame2_array[least_blurry_image_indx(frame2_array)]


                        #COMPLIANCE LOGIC
                        compliant = False
                        if most_common_goggle_detection == 0:
                            compliant = False  # no_mask
                        elif most_common_goggle_detection == 1:
                            compliant = True  # mask

                        
                        bordered_frame1 = draw_border(myFrame1, compliant, self.border_thickness)
                        bordered_frame2 = draw_border(myFrame2, compliant, self.border_thickness)
                        bordered_frame = np.hstack((bordered_frame1, bordered_frame2))
        

                        data = {
                                'zone_name': '1',
                                'crossing_type': 'coming',
                                'compliant' : str(compliant)
                            }

                        # NOW SEND IMAGE TO THE SERVER WITH DATA
                        #convert image to be ready to be sent
                        
                        success, encoded_image = cv2.imencode('.jpg', bordered_frame1)    
                        if success:
                            # Convert the encoded image to a byte array
                            image_bytes = bytearray(encoded_image)
                            # You can now use image_data like you did with f.read() 

                            # Send the image to the server
                            sendImageToServer(image_bytes, data, IP_address=self.server_IP)

                            print()
                            print("########### DETECTION MADE #############")
                            print(result_json)
                            print("########### END OF DETECTION #############")
                            print()
                        else:
                            raise ValueError("Could not encode the frame as a JPEG image")

                        if self.save:
                            self.save_frames([frame1_array, frame2_array])
                        #Finally clear detections array
                        detections_array = []
                        frame1_array = []
                        frame2_array = []

                zone_count = self.zones[0].current_count
                # Display frame
                if self.display:
                    cv2.imshow('ComboCam', frame)

                  # calculate and print the FPS
                # end_time = time.time()
                # fps = 1.0 / (end_time - start_time)
                # fps_list.append(fps)
                # print("FPS:", round(fps, 2))
                # print("Average FPS:", round(sum(fps_list) / len(fps_list), 2))
                # if len(fps_list) > 20:
                #     fps_list.pop(0)

            except Exception as e:
                print('frame unavailable', e)
                traceback.print_exc()

            if cv2.waitKey(1) == ord('q'):
                self.stop()
               
