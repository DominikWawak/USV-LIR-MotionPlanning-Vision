import cv2
import numpy as np
import config
import pafy
import networkx as nx
from math import sqrt


from roboflow import Roboflow
rf = Roboflow(api_key=config.apiKeySM2)
project = rf.workspace().project("saveme2")
model = project.version(2).model

# color in hue saturation value format

lower=np.array([90,150,20])
upper=np.array([138,255,255])


# Youtube Video
url = "https://youtu.be/POjC72UomUE"
video = pafy.new(url)
best = video.getbest(preftype="mp4")
video = cv2.VideoCapture(best.url)

# video = cv2.VideoCapture("res/people_water.mp4")


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
    success,img2 = video.read()
    image = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(image,lower,upper)

    

    # array for valid circle points
    valid_circles=[]
    startPoint=()
    endPoint=()
    # draw points 

   


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

        if(bounding_box):
            if(bounding_box['class']=="usv"):
                startPoint=start_point
            if(bounding_box['class']=="boat"):
                endPoint=start_point
       
            
    for i in pointGrid:
        if(mask[i[1],i[0]].sum()>0):
            valid_circles.append((i[0],i[1]))
            cv2.circle(img2, (i[0],i[1]), 10, (0,0,255), 2)


    if(startPoint and endPoint):
    
        valid_circles.append(startPoint)
        valid_circles.append(endPoint)
        # Create a graph object
        G = nx.Graph()

        # Add coordinate points as nodes
        # coordinate_points = [(25,125),(100,50),(150,50),(50,100),(100,100),(100,150),(150,150),(200,150),(200,200),(250,150),(300,100),
        # (350,100),(400,100),(400,150),(400,200),(450,100),(450,150),(450,200),(375,125)]
        for point in valid_circles:
            G.add_node(point)

        # Add edges between nodes with weights as Euclidean distance
        for i in range(len(valid_circles)):
            for j in range(i+1, len(valid_circles)):
                
                x1, y1 = valid_circles[i]
                x2, y2 = valid_circles[j]
                # distance = sqrt((x1 - x2)**2 + (y1 - y2)**2)
                # G.add_edge(coordinate_points[i], coordinate_points[j], weight=distance)
                if (abs(x1-x2)<=75 or abs(x1-x2)==0) and (abs(y1-y2)<=75 or abs(y1-y2)==0):
                    distance = sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    G.add_edge(valid_circles[i], valid_circles[j], weight=distance)


        # Find the shortest path between start point and end point using Dijkstra's algorithm
        # start = (25,125)
        # end = (375,125)
        # G.remove_edge(start, end)
        try:
            shortest_path = nx.dijkstra_path(G, startPoint, endPoint, weight='weight')

            print(shortest_path)

            for i in range(len(shortest_path)):
                if(i+1<len(shortest_path)):
                    cv2.line(img2, (shortest_path[i]), (shortest_path[i+1]), (0, 255, 0), thickness=3, lineType=8)



        except:
            print("no Path")



    # cv2.imshow("points",image)
    cv2.imshow("maks",mask)
    cv2.imshow("webcam",img)
    cv2.imshow("webcamspots",img2)

    cv2.waitKey(1)

