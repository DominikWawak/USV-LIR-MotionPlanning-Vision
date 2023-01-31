import cv2
import numpy as np
import config
import networkx as nx
from math import sqrt


# color in hue saturation value format

lower=np.array([90,150,20])
upper=np.array([138,255,255])


video = cv2.VideoCapture(0)

    

while True:
    success,img = video.read()
    image = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(image,lower,upper)


    # cv2.imshow("points",image)
    cv2.imshow("maks",mask)
    cv2.imshow("webcam",img)

    cv2.waitKey(1)

