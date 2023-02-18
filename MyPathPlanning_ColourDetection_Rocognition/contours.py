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
    mask = cv2.bitwise_not(mask)
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area>10000 and area<150000:
            print(area)
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(mask, (x-40,y-40),(x+w+40,y+h+40), (255,255,255),-1)
            




    # cv2.imshow("points",image)
    cv2.imshow("maks",mask)
    cv2.imshow("webcam",img)

    cv2.waitKey(1)

