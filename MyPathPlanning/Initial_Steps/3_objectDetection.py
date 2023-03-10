import cv2
import numpy as np
import config
import networkx as nx
from math import sqrt


from roboflow import Roboflow
rf = Roboflow(api_key=config.apiKeyPaper)
project = rf.workspace().project("usvlirpaper")
model = project.version(1).model

# color in hue saturation value format

lower=np.array([90,150,20])
upper=np.array([138,255,255])


video = cv2.VideoCapture(0)

if video.isOpened(): 
 
    
    width  = int(video.get(3))  # float `width`
    height = int(video.get(4))  # float `height`


    

while True:
    success,img = video.read()
    success,img2 = video.read()
    image = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

   



    prediction=model.predict(img, confidence=40, overlap=30).json()

    bounding_box=None
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

 


    
    cv2.imshow("webcam",img)
  

    cv2.waitKey(1)

