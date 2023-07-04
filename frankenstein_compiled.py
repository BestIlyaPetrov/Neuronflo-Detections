from flovision import laser_inference, entrance_inference, comms, video
import cv2
import argparse

# VIDEO RESOLUTION
video_res = [640, 480]

# NMS paparms
# conf_thres = 0.5
iou_thres = 0.7
agnostic_nms= True
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
    weights='bestmaskv5.pt',  # model path or triton URL,
    display_video = False,
    save_frames = False,
    new_boxes = False
    ):

    print_parameters()


    try:
        # Initialize the model
        inference_obj= laser_inference.LaserInferenceSystem(
            model_name = weights,
            video_res = video_res,
            border_thickness = border_thickness,
            display = display_video,
            save = save_frames,
            bboxes = new_boxes

        )
        #Run the model
        inference_obj.run(iou_thres, agnostic_nms)
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
    parser.add_argument('--weights', nargs='+', type=str, default='bestmaskv5.pt', help='model path')

    options = parser.parse_args()
    return options

if __name__ == "__main__":
    opt = parse_options()
    main(**vars(opt))
    