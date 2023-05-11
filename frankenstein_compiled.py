from flovision import inference, comms, video
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

#Region for goggles
w1 = 30 
h1 = 30
x1 = 50
y1 = 70

#Region for shoes
w2 = 80
h2 = 30
x2 = 50
y2 = 50

def print_parameters():
    print("VIDEO RESOLUTION: {} x {}".format(video_res[0], video_res[1]))
    print("NMS PARAMETERS:")
    print("  IOU threshold: {}".format(iou_thres))
    print("  Agnostic NMS: {}".format(agnostic_nms))
    print("IMAGE ANNOTATION PARAMETERS:")
    print("  Border thickness: {}".format(border_thickness))
    print("REGIONS FOR DETECTION:")
    print("  Goggles: x={}, y={}, w={}, h={}".format(x1, y1, w1, h1))
    print("  Shoes: x={}, y={}, w={}, h={}".format(x2, y2, w2, h2))


def main(
    display_video = False,
    save_frames = False
    ):

    print_parameters()

    # Prep the region coordinates
    CENTER_COORDINATES = [] #Center of the detection region as percentage of FRAME_SIZE
    WIDTH = []#% of the screen 
    HEIGHT = [] #% of the screen 

    CENTER_COORDINATES.append((x1 /2,y1)) #Center of the detection region as percentage of FRAME_SIZE
    WIDTH.append(w1 /2) #% of the 1st screen 
    HEIGHT.append(h1) #% of the 1st screen 

    CENTER_COORDINATES.append((50+( x2 /2),y2)) #Center of the detection region as percentage of FRAME_SIZE
    WIDTH.append(w2 /2) #% of the 2nd screen 
    HEIGHT.append(h2) #% of the 2nd screen 


    try:
        # Initialize the model
        inference_obj= inference.InferenceSystem(
            'bestmaskv5.pt',
            video_res,
            CENTER_COORDINATES,
            WIDTH,
            HEIGHT,
            border_thickness,
            display_video,
            save_frames

        )
        #Run the model
        inference_obj.run(iou_thres, agnostic_nms)
    except KeyboardInterrupt:
        print()
        print("Keyboard interrupt. Exiting peacefully")
        inference_obj.stop()

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--display-video', action='store_true', help='show video feed')
    parser.add_argument('--save-frames', action='store_true', help='save detected frames')
    options = parser.parse_args()
    return options

if __name__ == "__main__":
    opt = parse_options()
    main(**vars(opt))
    