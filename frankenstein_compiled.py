from flovision import inference, comms, video


# VIDEO RESOLUTION
video_res = [640, 480]

# NMS paparms
conf_thres = 0.5
iou_thres = 0.7
agnostic_nms= True
classes = None ## all classes are detected

#Image Annotation Params
border_thickness = 15

#Region for goggles
w1 = 30 
h1 = 100
x1 = 20
y1 = 50

#Region for shoes
w2 = 30 
h2 = 100
x2 = 20
y2 = 50

def print_parameters():
    print("VIDEO RESOLUTION: {} x {}".format(video_res[0], video_res[1]))
    print("NMS PARAMETERS:")
    print("  Confidence threshold: {}".format(conf_thres))
    print("  IOU threshold: {}".format(iou_thres))
    print("  Agnostic NMS: {}".format(agnostic_nms))
    print("  Classes: {}".format(classes))
    print("IMAGE ANNOTATION PARAMETERS:")
    print("  Border thickness: {}".format(border_thickness))
    print("REGIONS FOR DETECTION:")
    print("  Goggles: x={}, y={}, w={}, h={}".format(x1, y1, w1, h1))
    print("  Shoes: x={}, y={}, w={}, h={}".format(x2, y2, w2, h2))


def main():

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


    
    # Initialize the model
    inference_obj= inference.InferenceSystem(
        'bestmaskv5.pt',
        video_res,
        CENTER_COORDINATES,
        WIDTH,
        HEIGHT,
        border_thickness
    )
    #Run the model
    inference_obj.run()


if __name__ == "__main__":
    main()
