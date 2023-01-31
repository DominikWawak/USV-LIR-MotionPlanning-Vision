import cv2
import numpy as np
import config
import networkx as nx
from math import sqrt


# color in hue saturation value format

lower=np.array([90,150,20])
upper=np.array([138,255,255])


video = cv2.VideoCapture(0)

if video.isOpened(): 
 
    
    width  = int(video.get(3))  # float `width`
    height = int(video.get(4))  # float `height`

    pointGrid=[]

    for i in range(0,width,50):
        for j in range(0,height,50):
            pointGrid.append([i,j])
    # print(pointGrid)


    

while True:
    success,img = video.read()
    image = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(image,lower,upper)



    # array for valid circle points
    valid_circles=[]
    startPoint=()
    endPoint=()
    # draw points 

    for i in pointGrid:
        if(mask[i[1],i[0]].sum()>0):
            valid_circles.append((i[0],i[1]))
            cv2.circle(img, (i[0],i[1]), 10, (0,0,255), 2)




    # cv2.imshow("points",image)
    cv2.imshow("maks",mask)
    cv2.imshow("webcam",img)

    cv2.waitKey(1)

