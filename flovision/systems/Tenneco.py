from threading import Thread
import cv2
import time
import numpy as np
import torch, torchvision
import json
import datetime
import time

import supervision as sv
import traceback
import collections
from pathlib import Path

from ..bbox_gui import create_bounding_boxes, load_bounding_boxes
from ..video import draw_border, region_dimensions, vStream, least_blurry_image_indx, get_device_indices
from ..comms import sendImageToServer
from ..utils import get_highest_index, findLocalServer
from ..jetson import Jetson
from ..face_id import face_recog
from ..inference import InferenceSystem, Violation
from ..notifications import teleBot

from collections import Counter, defaultdict
from math import floor, ceil
import os
import datetime

"""
Model detects the following classes
0-goggles
1-no_goggles
2-soldering
3-hand

COCO dataset
0 - person
27 - tie
39 - bottle
42 - fork 
43 - knife
44 - spoon 
45 - bowl
56 - chair
57 - couch
59 - bed
60 - dining table
62 - tv
63 - laptop
64 - mouse
65 - remote
66 - keyboard
67 - cell phone
69 - oven
70 - toaster
71 - sink
72 - refridgerator
73 - book
76 - scissors
79 - toothbrush
"""

GOGGLES_CLASS = 1
NO_GOGGLES_CLASS = 0
BOOTS_CLASS = 2
NO_BOOTS_CLASS = 3

# GOGGLES_CLASS = 65 #remote
# NO_GOGGLES_CLASS = 66 #keyboard
# BOOTS_CLASS = 41 #cup
# NO_BOOTS_CLASS = 76 #scissors

TRACK_ID_KEEP_ALIVE = 1 # minutes

TOP_CAMERA_INDX = 0
BOTTOM_CAMERA_INDX = 1
class TennecoInferenceSystem(InferenceSystem):
    def __init__(self, *args, **kwargs) -> None:

        """
        self.frame_width = video_res[0]
        self.frame_height = video_res[1]
        self.frame_size = (self.frame_width, self.frame_height)
        self.display = display
        self.save = save
        self.annotate = annotate
        """

        # If need to overwrite a particular argument do the following. 
        # Let's say need to overwrite the 'model_directory' argument
        # kwargs['model_directory'] = 'my_new_directory_path'

        super().__init__(*args, **kwargs)
        self.initialize_system()

    def initialize_system(self) -> None:
        # List of lists, beacause for each camera feed, we have a list of violations we save to choose the prevalent one as jitter reduction technique (aka goggle, no_goggle, goggle -> goggle )
        self.array_for_frames = [[[] for _ in range(self.num_consecutive_frames)] for _ in range(len(self.cams))]
        self.detections_array = [[[] for _ in range(self.num_consecutive_frames)] for _ in range(len(self.cams))]
        self.violations_array = [[[] for _ in range(self.num_consecutive_frames)] for _ in range(len(self.cams))]
        self.consecutive_frames_cnt = [0 for _ in range(len(self.cams))] # counter for #of frames taken after trigger event


        self.violations_track_ids_array = [[] for _ in range(len(self.cams))]
        self.FrameProcessor = FrameProcessing(inference_system=self)
        self.violation_dictionary = [{} for _ in range(len(self.cams))]
        self.violation_to_server = [[] for _ in range(len(self.cams))]
        self.averaged_and_filtered_violations = [[] for _ in range(len(self.cams))]
        self.tracker_id_side_entered = [{} for _ in range(len(self.cams))]
        self.ready_to_send = [False for _ in range(len(self.cams))] #once N frames collected, the flag is true. once both flags are True - we send



    def findCenter(self, minX, minY, maxX, maxY):
        # Will find the center of a detection when given the 
        # min and max for the X and Y coordinates. 
        centerX = round(minX + (maxX - minX)/2)
        centerY = round(minY + (maxY - minY)/2)
        return centerX, centerY

    def tracker_id_side_entered_update(self) -> None:
        # List comprehension to dump old track ids
        # Will dump a track id if it's older than 1 min
        # key = tracker_id
        # value = ["L or R", datetime_obj] 
        if self.tracker_id_side_entered[self.camera_num]:
            print(f" ##### &&&&& tracker_side_entered dict is: {self.tracker_id_side_entered[self.camera_num].items()} ##### &&&&& ")
            self.tracker_id_side_entered[self.camera_num] = {key:value for key, value in self.tracker_id_side_entered[self.camera_num].items() if (datetime.datetime.now() - value[1]) < datetime.timedelta(minutes=TRACK_ID_KEEP_ALIVE)}

    def Update_Dictionary(self) -> None: #Ilya comment - this does not keep track of time tho?
        # Will update the violation_dictionary to contain the
        # most up to date violations. If there's a violation
        # that's over 10 minutes old, that violation will 
        # be deleted from the dictionary.  
        self.violation_dictionary[self.camera_num] = {key: value for key, value in self.violation_dictionary[self.camera_num].items() if len(value)}
        
    # def repeat_ids(self,list_of_track_ids):
    #     # Flatten the list
    #     flat_list = [item for sublist in list_of_track_ids for item in sublist]
    #     flat_list = [tuple(x) for x in flat_list]

    #     # Count occurrences
    #     count = Counter(flat_list)
        
    #     # Get numbers that appear at least floor(len(a)/2) times
    #     return [num for num, freq in count.items() if freq >= floor(len(list_of_track_ids)/2)]


    def repeat_ids(self, list_of_track_ids):
        if self.debug:
            print(f'list_of_track_ids = {list_of_track_ids}')
        id_counts = defaultdict(int)    # Dictionary to store occurrences of each ID
        frame_counts = defaultdict(set) # Dictionary to store which frames the ID appeared in
        
        # Iterate over the list_of_track_ids and the IDs in each frame
        for frame_num, frame in enumerate(list_of_track_ids):
            for id_ in frame:
                id_counts[id_] += 1
                frame_counts[id_].add(frame_num)
        if self.debug:
            print(f'id_counts = {id_counts}')
            print(f'frame_counts = {frame_counts}')




    def process_violations_array_top_cam(self, violations_array):
        track_id_counts = Counter()

        # Collect class_ids for each track_id
        class_ids_per_track = {}

        for sublist in violations_array:
            for detection in sublist:
                track_id = detection['track_id']
                track_id_counts[track_id] += 1
                class_id = detection['class_id']
                if track_id not in class_ids_per_track:
                    class_ids_per_track[track_id] = []
                class_ids_per_track[track_id].append(class_id)

        num_sublists = len(violations_array)
        filtered_array = []

        for sublist in violations_array:
            new_sublist = []
            # Find the avg detection between (goggles/no_goggles) and set every detection to that class_id
            for detection in sublist:
                track_id = detection['track_id']
                if track_id_counts[track_id] > num_sublists // 2:
                    # Calculate the average class_id for this track_id
                    most_common_class_id = Counter(class_ids_per_track[track_id]).most_common(1)[0][0]
                    # Update the detection with the newly found class_id that is prevalent (most_common) in the array
                    detection['class_id'] = most_common_class_id
                    new_sublist.append(detection)

            if new_sublist:
                filtered_array.append(new_sublist)

        return filtered_array

    def process_violations_array_bottom_cam(self, violations_array):
        all_class_ids = []

        for sublist in violations_array:
            for detection in sublist:
                all_class_ids.append(detection['class_id'])

        if not all_class_ids:
            return []

        most_prevalent_class_id = Counter(all_class_ids).most_common(1)[0][0]

        # Update all class_ids in the violations_array
        for sublist in violations_array:
            for detection in sublist:
                detection['class_id'] = most_prevalent_class_id

        return violations_array


    # re-write trigger event to give true if there are any detections in any zone in self.zone_polygons
    def trigger_event(self) -> bool:
        bool_trigger = False
        if self.detection_trigger_flag[self.camera_num]:
            return bool_trigger
        
        # check the top camera for detections
        if self.camera_num == TOP_CAMERA_INDX:
            for zone in self.zone_polygons:
                if zone.camera_id == TOP_CAMERA_INDX and zone.PolyZone.current_count > zone.last_count:
                    # update the last_count to the current count
                    # so that next time we can trigger upon new detections in a zone
                    bool_trigger = True
                zone.last_count = zone.PolyZone.current_count

        # If objects were detected from the top camera, set the bottom camera trigger flag as well
        if bool_trigger:
            self.detection_trigger_flag[BOTTOM_CAMERA_INDX] = True    

        return bool_trigger

    def annotate_violations(self, frame, detections) -> list:
        """
        NOT TESTED
        """
        # Use this function to annotate the frame's
        # valid violations.   
        violations = self.violation_to_server[self.camera_num]
        least_blurry_indx = least_blurry_image_indx(self.array_for_frames[self.camera_num])
        frame = self.array_for_frames[self.camera_num][least_blurry_indx]
        detections = self.detections_array[self.camera_num][least_blurry_indx]
        # violation = [person_index, soldering_iron_index, camera_index, violation_code, track_id]
        # violations = [violation, violation, ...]
        person_text = 'No Goggles'
        shoes_text = 'Wrong Shoes'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)  # Red color in BGR format
        box_color = (0, 0, 255)
        line_thickness = 2
        font_thickness = 2

        # violations = [[class_id, detection_index, detection_track_id],
        #               [class_id, detection_index, detection_track_id], ...]

        for violation in violations:
            # Each iteration will text annotate a full violation
            # onto the frame. Each annotation will have a [int] 
            # at the end of the text annotation from the frame.
            # This is to indicate which detection is with which
            # that caused a violation 
            class_id = violation[0]
            detection_idx = violation[1]
            Xmin, Ymin, Xmax, Ymax = detections.xyxy[detection_idx]
            positions = [(int(Xmin), int(Ymin)), (int(Xmax), int(Ymax))]
            text_position = (int(Xmin), int(Ymin)-5)
            annotation_text = person_text if class_id == 1 else shoes_text
            # For the text background
            # Finds space required by the text so that we can put a background with that amount of width.
            (w, h), _ = cv2.getTextSize(annotation_text, font, font_scale, font_thickness)
            # Text background
            frame = cv2.rectangle(frame, (Xmin, Ymin - 20), (Xmin + w, Ymin), box_color, -1)
            # Label text
            frame = cv2.putText(frame, annotation_text, text_position, font, font_scale, font_color, font_thickness)
            # Bounding box
            frame = cv2.rectangle(frame, positions[0], positions[1], box_color, line_thickness)

        return frame

    def trigger_action(self) -> None:
        """
        NOT TESTED
        """

        """
        NOTES
        1) Ask Ilya about hard coding the camera's index to be able to control
           which camera is getting consecutive frames
        """
        
        # After N consecutive frames collected, check what was the most common detection for each object
        corrected_violations = []
        least_blurry_indx = None


        # The following 3 arrays are appened in parallel

        # Append detections from self.camera_num source to the array for detection jitter reduction
        # print(f"self.detections_array is {self.detections_array}")
        # print(f"using self.camera_num= {self.camera_num} and self.consecutive_frames_cnt[self.camera_num] = {self.consecutive_frames_cnt[self.camera_num]}")
        self.detections_array[self.camera_num][self.consecutive_frames_cnt[self.camera_num]] = self.detections

        # if self.consecutive_frames_cnt[self.camera_num] > 1: #if cnt==1, we just saved the self.violations in the trigger_event(), no need to do it again
        self.violations = self.FrameProcessor.NewDetections(self.detections)

        # Append violations found in the detections above
        self.violations_array[self.camera_num][self.consecutive_frames_cnt[self.camera_num]]= self.violations
        # print(f'[CAMERA {self.camera_num}] Appending {self.violations} \n to the violations_array and it is now:')
        # for i, violations in enumerate(self.violations_array[self.camera_num]):
        #     print(f'   [CAMERA {self.camera_num}] violations in frame {i} = {violations}')
        
        # Append an image from self.camera_num source to the array for least blurry choice
        self.array_for_frames[self.camera_num][self.consecutive_frames_cnt[self.camera_num]] = self.captures_HD[self.camera_num]
        # self.array_for_frames[self.camera_num][self.consecutive_frames_cnt[self.camera_num]] = self.annotated_frame
        # self.array_for_frames is an array of arrays of images

        if self.debug:
            # Need to print out the contents to check if images do get saved in each list
            print(f'[CAMERA {self.camera_num}] The shape of the array_for_frames[{[self.camera_num]}] is {np.shape(self.array_for_frames[self.camera_num])}')
            # now print the shape of each element inside self.array_for_frames[self.camera_num]
            
            for i, frame in enumerate(self.array_for_frames[self.camera_num]):
                print(f'   [CAMERA {self.camera_num}] The shape of the frame {i} is {np.shape(frame)}')

        # Count how many images we already took
        self.consecutive_frames_cnt[self.camera_num] += 1

        if  self.consecutive_frames_cnt[self.camera_num] >= self.num_consecutive_frames:

            # Reset the consecutive frames count
            self.consecutive_frames_cnt[self.camera_num] = 0

            # Reset the trigger flag as well so we can have another trigger action
            self.detection_trigger_flag[self.camera_num] = False

            # When both the top and bottom camera collect enough frames, we send the data to the server
            self.ready_to_send[self.camera_num] = True

    


            ####### COMMENTS FOR MYSELF #########
            # AT THIS POINT violations_array is a list of detections, whose track_ids initially appeared in the further zone
            # array_for_frames is a list of frames that correspond to the detections in violations_array
            # So, those people that are walking toward the camera
            # Now we need to parse through that list of detctions and ########### important ###########
            # for each track_id determine what was the most prevalent violation ########### important ###########
            # We then make the compliance logic happen based on detections in both cameras
            # (ig the bottom camera logic is just if only "boots" is preent - we good)
            # Then we send that together 
            if self.debug:
                print(f'[CAMERA {self.camera_num}] violations_array = {self.violations_array[self.camera_num]}\n')
                # Now print individual detections in each frame
                for i, violations in enumerate(self.violations_array[self.camera_num]):
                    print(f'[CAMERA {self.camera_num}] violations in frame {i} = {violations}')
                    # Now print individual detections in each frame and numerate them and show whcih frame they belong to
                    for j, violation in enumerate(violations):
                        print(f'   [CAMERA {self.camera_num}] violation {j} in frame {i} = {violation}')
                    print()

            if self.camera_num == TOP_CAMERA_INDX:
                self.averaged_and_filtered_violations[self.camera_num] = self.process_violations_array_top_cam(self.violations_array[self.camera_num])
                # In case there's overlap and there's more than 1 person, pick out the frame with most detections (yet send the image of least blurry one still)
                if len(self.averaged_and_filtered_violations[self.camera_num]) > 0:
                    self.violation_to_server[self.camera_num] = max(self.averaged_and_filtered_violations[self.camera_num], key=len)# Find the longest sublist in the filtered result
                    if self.debug:
                        print(f"\n %% len(self.averaged_and_filtered_violations[{self.camera_num}]) = {len(self.averaged_and_filtered_violations[self.camera_num])}\n")
                else:
                    if self.debug:
                        print(f"\n######## No detections to send from camera {self.camera_num} - it was all a glitch #####\n")

    
            else:
                ## ADD BOTTOM CAMERA LOGIC HERE
                self.averaged_and_filtered_violations[self.camera_num] = self.process_violations_array_bottom_cam(self.violations_array[self.camera_num])
                if len(self.averaged_and_filtered_violations[self.camera_num]) > 0:
                    # FIX THIS LOGIC to pick least blurry, but only the one with violations
                    self.violation_to_server[self.camera_num] = max(self.averaged_and_filtered_violations[self.camera_num], key=len)# Find the longest sublist in the filtered result

                    #print the len:
                    if self.debug:
                        print(f"\n %% len(self.averaged_and_filtered_violations[{self.camera_num}]) = {len(self.averaged_and_filtered_violations[self.camera_num])}\n")
                else:
                    if self.debug:
                        print(f"\n######## No detections to send from camera {self.camera_num} - it was all a glitch #####\n")
            
            # Finds the least blurry image's index
            
            """
            # This records the union of track_ids of the least blurry image and which 
            # ids are followed in multiple frames 
            for violation in self.violations_array[self.camera_num][least_blurry_indx]:
                # For this code to work, note that the last index of violation must
                # be the tracker id of that set of detections
                if violation[-1] in violating_ids_list:
                    corrected_violations.append(violation)
            """

            if all(self.ready_to_send) and any(len(cam_violations) > 0 for cam_violations in self.violation_to_server[TOP_CAMERA_INDX]):
                self.system_send()




            # Now reset the arrays if both are done collecting the N frames
            if all(self.ready_to_send):
                self.array_for_frames = [[[] for _ in range(self.num_consecutive_frames)] for _ in range(len(self.cams))]
                self.detections_array = [[[] for _ in range(self.num_consecutive_frames)] for _ in range(len(self.cams))]
                self.violations_array = [[[] for _ in range(self.num_consecutive_frames)] for _ in range(len(self.cams))]
                self.violation_to_server = [[] for _ in range(len(self.cams))]
                self.averaged_and_filtered_violations = [[] for _ in range(len(self.cams))]
                self.consecutive_frames_cnt = [0 for i in range(len(self.cams))] # counter for #of frames taken after trigger event
                #clear the ready_to_send flag
                self.ready_to_send = [False for _ in range(len(self.cams))]
                if self.debug:
                    print("\n######## CLEARED MAIN ARRAYS ########")
                    #Print how logn it took to execute using info from self.trigger_action_start_timer
                    print(f"\n######## It took {datetime.datetime.now() - self.trigger_action_start_timer} seconds to execute trigger_action ########\n")

  
    def system_send(self):


            # zone_name = request.POST.get('zone_name', '')
            # crossing_type = request.POST.get('crossing_type', '')
            # compliant = request.POST.get('compliant', '')
            # ppe_status = request.POST.get('ppe_status', '')



        

        # Define the violations to be sent to the server
        # if violation_to_server[0] contains at least 1 class_id=1, then goggle_status = False
        goggle_status = False if any([violation["class_id"]==NO_GOGGLES_CLASS for violation in self.violation_to_server[TOP_CAMERA_INDX]]) else True
        shoe_status = False if any([violation["class_id"]==NO_BOOTS_CLASS for violation in self.violation_to_server[BOTTOM_CAMERA_INDX]]) else True
        # Now print in detail how goggle_status is compiled. Print violation["class_id"] in each violation
        

        #print every class id in each violation in violation_to_server
        if self.debug:
            print("\n######## TOP CAMERA VIOLATION - checking goggles ########\n")
            for violation in self.violation_to_server[TOP_CAMERA_INDX]:
                print(f"\n######## violation class_id = {violation['class_id']} ########")
            print(f"\n ######## goggle_status = {goggle_status} ########\n")

            print("\n ######## BOTTOM CAMERA VIOLATION - checking shoes ########\n")
            for violation in self.violation_to_server[BOTTOM_CAMERA_INDX]:
                print(f"\n ######## violation class_id = {violation['class_id']} ########\n")
            print(f"\n ######## shoe_status = {shoe_status} ########\n")





        

        ppe_status = {"goggles": str(goggle_status),
                      "shoes":str(shoe_status)}
        ppe_status = json.dumps(ppe_status)

        # if either goggle_status or shoe_status is False, then compliance_status = False
        compliance_status = False if not (goggle_status and shoe_status) else True

        # Save the image locally for further model retraining
        if self.save:
            self.save_frames(frame, self.camera_num)

        # Annotate the violations #Needs to be tested
        # self.frame_with_violation = self.annotate_violations()

        #  Compliance Logic
        # img_to_send = frame

        timestamp_to_send = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        data = {
            'timestamps': timestamp_to_send, # We only need a timestamp
            'ppe_status': ppe_status,
            'compliant': compliance_status,
            'zone_name': 'main_entrance',
            'crossing_type': 'coming',
            'event_type' : 'crossing'
        }

        # data['ppe_status'] = ','.join([f"{k}:{v}" for k, v in data['ppe_status'].items()])

        # nicely print the data that is about to be sent - make it distinguishable with spaces and hastags
        # print the data by key pairs
        if self.debug:
            print("\n##########################################################################")
            print( f"###################### Sending data to the server ########################")
            print(  "##########################################################################\n")
            for key, value in data.items():
                print(f"{key} : {value}")                


        # Find the frame with the least blurry index
        frames_to_send = []
        for frame_arr in self.array_for_frames:
            least_blurry_indx = least_blurry_image_indx(frame_arr)
            frames_to_send.append(frame_arr[least_blurry_indx])
            # annotated_frame = self.annotate_violations(frame, violation_boxes) #Needs to be FIXED #TODO
        frame_to_send = np.hstack((frames_to_send[TOP_CAMERA_INDX], frames_to_send[BOTTOM_CAMERA_INDX]))
        


        
        link = "api/event_update"
        sendImageToServer(frame_to_send, data, self.server_IP, link)

class FrameProcessing():
    # A class for processing detections found in a frame  
    def __init__(self, inference_system) -> None:
        # self.detections = None
        self.system = inference_system
############################################# NEED TO RE-WRITE THIS #############################################
    def NewDetections(self, current_detections) -> list:
        # Will be used to update the detections and
        # returns the relevant detection data 
        # relating to violations detected so far

        
        return self.violation_creator(current_detections)

    
    def violation_creator(self, current_detections) -> list:
        track_id = None
        head_class_set = [4]
        
        filtered_detections = []
        goggle_classes_set = [GOGGLES_CLASS, NO_GOGGLES_CLASS] #goggles, no_goggles
        shoe_classes_set = [BOOTS_CLASS, NO_BOOTS_CLASS] #safe_shoes, not_safe_shoes
        
        # Clear any track_ids that have been there longer than TRACK_ID_KEEP_ALIVE (aka 1 min)
        self.system.tracker_id_side_entered_update()
        if self.system.debug:
            print(f"\n\n ######## Processing camera {self.system.camera_num} ########\n\n")

        if self.system.camera_num == BOTTOM_CAMERA_INDX: #tentatively not trying to track ENTERING/EXITING in the bottom camera because expecting bytetracker to glitch on fast movements of the feet
            if len(current_detections)==0:
                return []
            
            shoe_zone_detections = current_detections[self.system.masks[2] & np.isin(current_detections.class_id, shoe_classes_set)]
            return [{"bbox":detection[0], "confidence":detection[-3], "class_id":detection[-2], "track_id":detection[-1]} for detection in shoe_zone_detections]
           
        
        else:# self.system.camera_num == TOP_CAMERA_INDX:

            if len(current_detections)==0:
                return []
            # There are two zones, the further zone and the closer zone
            # People enter the frame through the further zone when they walk in
            # and they enter the frame through the closer zone when they walk out
            # Let's filter out goggle/no_goggle detections in each zone
            # Then save them to a dict to remember where a particular person entered initially (using track_id of course)
            ### Assuming masks[0] is the further zone, masks[1] is the closer zone ###

            # checking goggles/no_goggles in the two zone in the top camera
            further_zone_detections = current_detections[self.system.masks[0] & np.isin(current_detections.class_id, goggle_classes_set)]
            closer_zone_detections = current_detections[self.system.masks[1] & np.isin(current_detections.class_id, goggle_classes_set)]
            
            # if self.system.debug:
            #     #Print the detections in each zone
            #     print(f'\nFURTHER_zone_detections = {further_zone_detections}\n')
            #     print(f'\nCLOSER_zone_detections = {closer_zone_detections}\n')

            # detection[-1] is detection's track_id
            # detection[-2] is detection's class_id
            # detection[-3] is detection's confidence


            # Check if the person has already been detected ENTERING or EXITING
            # 1) We take all the detections in the further zone and check if they have been detected before
            # 2) If those track_ids initially started off in the closer zone, these detections just get dropped
            # 3) If those track_ids initially started off in the further zone, these detections get added to the violations list
            # 4) We take all the detections in the closer zone and check if they have been detected before
            # 5) If those track_ids initially started off in the further zone, these detections get added to the violations list
            # 6) If those track_ids initially started off in the closer zone, these detections just get dropped
            filtered_detections.append(self.process_detections(further_zone_detections, "ENTERING")) #(detections, zone_status_to_add_if_new)
            filtered_detections.append(self.process_detections(closer_zone_detections, "EXITING")) #(detections, zone_status_to_add_if_new)

            # Cleaned up detections. Every track_id that has/had status "leaving" has been cleared from the memory
            merged_detections = filtered_detections[0] + filtered_detections[1]


            return merged_detections
                    

    # Only returns goggle/no_goggle detections for people who are entering the facility
    def process_detections(self, zone_detections, zone_status):
        filtered_detections = []
        for detection in zone_detections:
            track_id = detection[-1]
            if self.system.debug:
                print(f'{zone_status} zone detection track_id = {track_id}')

            if (track_id is not None) and (track_id not in self.system.tracker_id_side_entered[self.system.camera_num]):
                self.system.tracker_id_side_entered[self.system.camera_num][track_id] = [zone_status, datetime.datetime.now()]
                if self.system.debug:
                    print(f"######### Adding track_id: {track_id} as {zone_status} to the memory dict ############")

            elif track_id is not None:
                if self.system.debug:
                    print(f"Track_id:{track_id} is already in the memory. Person was {self.system.tracker_id_side_entered[self.system.camera_num][track_id][0]}")  # ENTERING/EXITING
            
            # Now refer back to the newly updated dict to see if the person was ENTERING or EXITING initially
            if self.system.tracker_id_side_entered[self.system.camera_num][track_id][0] == "ENTERING":
                # filtered_detections.append(detection)
                filtered_detections.append({"bbox":detection[0], "confidence":detection[-3], "class_id":detection[-2], "track_id":detection[-1]})

                if self.system.debug:
                    print(f"######### Therefore, adding track_id: {track_id} to the detections list because person was ENTERING ############")
        return filtered_detections