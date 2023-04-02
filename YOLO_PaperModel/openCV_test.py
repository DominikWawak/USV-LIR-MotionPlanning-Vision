import cv2
import numpy as np


cap = cv2.VideoCapture(0)

# load model
net = cv2.dnn.readNetFromONNX('/Users/dominikwawak/Documents/FinalYear/Project/motionplanningStuff/USV-LIR-MotionPlanning-Vision/MyPathPlanning_ColourDetection_Rocognition/YOLO_Files/yolov5/runs/train/exp3/weights/best.onnx')
file=open('/Users/dominikwawak/Documents/FinalYear/Project/motionplanningStuff/USV-LIR-MotionPlanning-Vision/MyPathPlanning_ColourDetection_Rocognition/YOLO_Files/yolov5/usvlirpaper-2/coco.txt','r')
classes=file.read().split('\n')
print(classes)


while True:
    frame = cap.read()[1]

    if frame is None:
        break

    
    
    # cv2.dnn.blobFromImage() is a function used to create a 4-dimensional blob from an image. It takes the following parameters:

    # frame: The image to be converted into a blob.
    # 1/255: The scale factor used to normalize the image.
    # (640,640): The size of the image.
    # [0,0,0]: The mean subtraction values for each channel.
    # 1: The value used to scale the image.
    # swapRB: A boolean value indicating whether to swap the first and last channels of the image.
    # crop: A boolean value indicating whether to crop the image.

    blob=cv2.dnn.blobFromImage(frame,1/255,(640,640),[0,0,0],swapRB=True,crop=False)
    net.setInput(blob)
    detections = net.forward()[0]
    # predicts 25200 boxes or detections, each box has 8 entries
    print(detections.shape)

    #cx cy w h conf class 8 class scores
    # class+ids, confidence, bounding box

    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]

    #scale
    frame_width, frame_height = frame.shape[1], frame.shape[0]
    x_scale = frame_width / 640
    y_scale = frame_height / 640

    for i in range(rows):
        row=detections[i]
        confidence=row[4]
        #threshold
        if confidence > 0.5:
            classes_score=row[5:]
            ind=np.argmax(classes_score)
            if classes_score[ind] > 0.5:
                classes_ids.append(ind)
                confidences.append(float(confidence))
                x_center,y_center,width,height=row[0:4]
                x=int((x_center-width/2)*x_scale)
                y=int((y_center-height/2)*y_scale)
                width=int(width*x_scale)
                height=int(height*y_scale)
                boxes.append([x,y,width,height])
    
    #remove extra boxes
    indices= cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
    
    for i in indices:
        x,y,width,height=boxes[i]
        label=classes[classes_ids[i]]
        conf=confidences[i]
        text=label+":"+str(round(conf,2))
        cv2.rectangle(frame,(x,y),(x+width,y+height),(0,255,0),2)
        cv2.putText(frame,text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)


    cv2.imshow('frame', frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

