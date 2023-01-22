import cv2
import numpy as np
import config

from roboflow import Roboflow
rf = Roboflow(api_key=config.apiKey)
project = rf.workspace().project("usvlirpaper")
model = project.version(1).model

# color in hue saturation value format

lower=np.array([90,150,20])
upper=np.array([138,255,255])


video = cv2.VideoCapture(0)

if video.isOpened(): 
 
    
    width  = int(video.get(3))  # float `width`
    height = int(video.get(4))  # float `height`

    pointGrid=[]

    for i in range(0,width,100):
        for j in range(0,height,100):
            pointGrid.append([i,j])
    # print(pointGrid)


    

while True:
    success,img = video.read()
    image = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(image,lower,upper)


    # draw points 

    for i in pointGrid:
        if(mask[i[1],i[0]].sum()>0):
            cv2.circle(img, (i[0],i[1]), 20, (0,0,255), 2)

        




    prediction=model.predict(img, confidence=40, overlap=30).json()


    for i in range(0,3,1):
        try:

            print(prediction['predictions'][i]['class'])


            bounding_box=prediction['predictions'][i]
            x0 = bounding_box['x'] - bounding_box['width'] / 2
            x1 = bounding_box['x'] + bounding_box['width'] / 2
            y0 = bounding_box['y'] - bounding_box['height'] / 2
            y1 = bounding_box['y'] + bounding_box['height'] / 2

            start_point = (int(x0), int(y0))
            end_point = (int(x1), int(y1))
            cv2.rectangle(img, start_point, end_point, color=(0,0,0), thickness=2)

            cv2.putText(
            img, # PIL.Image object to place text on
            bounding_box['class'],#text to place on image
            (int(x0), int(y0)+10),#location of text in pixels
            fontFace = cv2.FONT_HERSHEY_SIMPLEX, #text font
            fontScale = 0.6,#font scale
            color = (255, 255, 255),#text color in RGB
            thickness=2#thickness/"weight" of text
    )
        except:
            print("Nothing found")


    # cv2.imshow("points",image)
    cv2.imshow("maks",mask)
    cv2.imshow("webcam",img)

    cv2.waitKey(1)

