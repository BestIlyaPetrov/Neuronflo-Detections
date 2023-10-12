# from flovision import laser_inference, entrance_inference, comms, video, inference
from flovision.systems import Entrance, Envision, LaserCutter, FaceRecognition, Tenneco
import cv2
import argparse

# VIDEO RESOLUTION
video_res = [640, 360]

# NMS paparms
# conf_thres = 0.5
iou_thres = 0.7
agnostic_nms= False
# classes = None ## all classes are detected

#Image Annotation Params
border_thickness = 15


def print_parameters():
    print("VIDEO RESOLUTION: {} x {}".format(video_res[0], video_res[1]))
    print("NMS PARAMETERS:")
    print("  IOU threshold: {}".format(iou_thres))
    print("  Agnostic NMS: {}".format(agnostic_nms))
    print("IMAGE ANNOTATION PARAMETERS:")
    print("  Border thickness: {}".format(border_thickness))



def main(
    weights='custom_models/bestmaskv5.pt',  # model path or triton URL,
    display_video = False,
    save_frames = False,
    new_boxes = False,
    server_IP = 'local',
    annotate_raw = False,
    annotate_violation = True,
    debug = False,
    record = False
    ):

    print_parameters()


    try:
        # Initialize the model
        inference_obj= Tenneco.TennecoInferenceSystem(
            model_name = weights,
            video_res = video_res,
            # border_thickness = border_thickness, # we are not drawing borders anymore
            display = display_video,
            save = save_frames,
            bboxes = new_boxes,
            num_devices=2,
            model_type="custom",
            model_directory="./",
            model_source="local",
            detected_items=["goggles","no_goggles","boots","no_boots"], #this is nbot currently being used
            server_IP = server_IP,
            annotate_raw = annotate_raw,
            annotate_violation = annotate_violation,
            debug=debug,
        )
        inference_obj.run(iou_thres=iou_thres, agnostic_nms=agnostic_nms)

        # inference_obj = inference.EntranceInferenceSystem(
        print("Model name is:", weights)
        """
        inference_obj = Envision.EnvisionInferenceSystem(
            model_name = weights, 
            video_res = video_res, 
            border_thickness = border_thickness, 
            display = display_video, 
            save = save_frames, 
            bboxes = new_boxes,
            num_devices=2,
            model_type="custom",
            model_directory="./",
            model_source="local",
            detected_items=["goggles","no_goggles","soldering","hand"],
            server_IP = server_IP,
            annotate_raw = annotate_raw,
            annotate_violation = annotate_violation,
            debug=debug,
            record=record
            )
        """
        #Run the model
        # inference_obj.run(iou_thres, agnostic_nms)
    except KeyboardInterrupt:
        print()
        print("Keyboard interrupt. Exiting peacefully")
        if inference_obj:
            inference_obj.stop()

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--display-video', action='store_true', help='show video feed')
    parser.add_argument('--save-frames', action='store_true', help='save detected frames')
    parser.add_argument('--new-boxes', action='store_true', help='create new bounding boxes')
    parser.add_argument('--weights', type=str, default='custom_models/bestmaskv5.pt', help='model path')
    parser.add_argument('--server_IP', type=str, default='local', help='IP of the server the images are being sent to')
    parser.add_argument('--annotate-raw', action='store_true', help='return images with detection boxes')
    parser.add_argument('--annotate-violation', action='store_true', help='annotate the violation')
    parser.add_argument('--debug', action='store_true', help='verbose mode')
    parser.add_argument('--record', action='store_true', help='records the last 5 seconds')


    options = parser.parse_args()
    return options

if __name__ == "__main__":
    opt = parse_options()
    main(**vars(opt))
    
