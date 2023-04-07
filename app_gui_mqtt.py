
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from cap_from_youtube import cap_from_youtube
import cv2
import cvzone
import numpy as np
import config
import networkx as nx
from math import sqrt
import time
import threading
from roboflow import Roboflow
import math 
import paho.mqtt.client as mqtt
from pymavlink import mavutil
import time 
import json



ack_msg=""
boat_ready_msg=""
path_finished=False
boat_heading=""
simultaion_mode=False
usv_simulation = False
start_path_planning = False
ignore_water = False
local_model=False




#****************************************************************************************************
#*************MQTT Setup************************************************************************

# Callback when the connection to the broker is established
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    # Subscribe to a channel
    client.subscribe("test/res")

# Callback when a message is received
def on_message(client, userdata, msg):
    global ack_msg
    print("Received message on channel " + msg.topic + ": " + str(msg.payload))
    m_decode = str(msg.payload.decode("utf-8", "ignore"))
    m_decode = m_decode.replace(",", ':')
    print("data Received", m_decode.split(":")[1])
    message_decoded = m_decode.split(":")[1]
    print("message_decoded", message_decoded)
    if(message_decoded=='"ack"'):
        ack_msg=m_decode.split(":")[1]
    elif(message_decoded=="boat_ready"):
        boat_ready_msg=m_decode.split(":")[1]
    elif(message_decoded=="N"):
        boat_heading=m_decode.split(":")[1]

# Create a new MQTT client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Connect to the Beebotte broker
client.username_pw_set("9h1rq3Hf5cxTeQcb2yTYK3N6", "m33rs3IoJWxT9eX01hoxTLIDfLtq3EWN")
client.connect("mqtt.beebotte.com", 1883, 60)


# # Publish the payload to a channel
# client.publish("test/res", 'Hello World')

# Wait for incoming messages

tmqtt=threading.Thread(target=client.loop_forever)
tmqtt.start()
print("Mqtt thread started")

time.sleep(5)
#****************************************************************************************************
#*************MQTT Setup************************************************************************


def getCompassFromBoat():
   
        # Start a connection listening on a UDP port
    the_connection = mavutil.mavlink_connection('udpin:localhost:14445')

    # Wait for the first heartbeat 
    #   This sets the system and component ID of remote system for the link
    the_connection.wait_heartbeat()
    print("Heartbeat from system (system %u component %u)" % (the_connection.target_system, the_connection.target_component))

    while 1:
        msg = the_connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        #Print the compass value
        print("Global Position: %s" % int(msg.hdg/100))
        client.publish("test/compass", int(msg.hdg/100))
        time.sleep(0.5)


compassThread = threading.Thread(target=getCompassFromBoat)
compassThread.start()

time.sleep(5)

print("compass thread started")

def motion_recognitionThread(option,mode):
    global local_model
    # key,projectName,version
    # rf = Roboflow(key)
    # project = rf.workspace().project(projectName)
    # model = project.version(version).model

    

    print(mode)
    
    if(mode=="paper"):
        rf = Roboflow(api_key=config.apiKeyPaper)
        project = rf.workspace().project("usvlirpaper")
        model = project.version(1).model
    elif(mode=="realLife"):
        rf = Roboflow(api_key=config.apiKeySM2)
        project = rf.workspace().project("saveme2")
        model = project.version(2).model
    elif(mode=="search_and_rescue"):
        rf = Roboflow(api_key="vcq6WDRWgu2bLiH4FVT5")
        project = rf.workspace().project("usv_lir_search_and_rescue")
        model = project.version(3).model
    elif(mode=="LOCAL-paper-version1"):
        # load model
       
        local_model=True
        net = cv2.dnn.readNetFromONNX('/Users/dominikwawak/Documents/FinalYear/Project/motionplanningStuff/USV-LIR-MotionPlanning-Vision/YOLO_PaperModel/yolov5/runs/train/exp3/weights/best.onnx')
        file=open('/Users/dominikwawak/Documents/FinalYear/Project/motionplanningStuff/USV-LIR-MotionPlanning-Vision/YOLO_PaperModel/yolov5/usvlirpaper-2/coco.txt','r')
        classes=file.read().split('\n')
        print(classes)
    elif(mode=="LOCAL-real-version1"):
        # load model
        
        local_model=True
        net = cv2.dnn.readNetFromONNX('/Users/dominikwawak/Documents/FinalYear/Project/motionplanningStuff/USV-LIR-MotionPlanning-Vision/YOLO_RealModel/yolov5/runs/train/exp4/weights/best.onnx')
        file=open('/Users/dominikwawak/Documents/FinalYear/Project/motionplanningStuff/USV-LIR-MotionPlanning-Vision/YOLO_RealModel/yolov5/SEARCH_AND_RESCUE_SMALL_LOCAL-1/coco.txt','r')
        classes=file.read().split('\n')
        print(classes)


    if(option==1):
        video = cv2.VideoCapture(0)
    elif(option==2):
        fn= askopenfilename()
        print("user chose", fn)
        video = cv2.VideoCapture(fn)
    elif(option==3):
        url = app.stream3Text.get()
        if 'youtube' in url:
            video=cap_from_youtube(url)
        else:
            video = cv2.VideoCapture(url)
            #http://192.168.45.159:6868/screen_stream.mjpeg
        # best = video.getbest(preftype="mp4")
        # video = cv2.VideoCapture(best.url)
        # TEST VIDEO FOR PERSON https://www.youtube.com/watch?v=5n7ZNLvegBo
        # TEST VIDEO KAYAK https://www.youtube.com/watch?v=KLmXxadpTtQ

    width=0
    height=0

    usv_width_cm=8 #cm
    usv_width_pixels=0
    pixels_per_cm=0

    # video = cv2.VideoCapture(0)
    if video.isOpened(): 
    
        
        width  = int(video.get(3))  # float `width`
        height = int(video.get(4))  # float `height`

        pointGrid=[]

        for i in range(0,width,50):
            for j in range(0,height,50):
                pointGrid.append([i,j])
       

    previous_shortest_path=[]
    valid_circles_path_found=[]
    point_drift=(0,0)
    startPoint_prev=(-5000000,-500000)
    endPoint_prev=(-5000000,-500000)
    validPointCount=0
    path_found=False 
    shortest_path=[]
    directions=[]
    distances=[]
    G_found = nx.Graph()
    




    # 
    # Loop
    # 


    # Create the person simulated image
    # load the overlay image. size should be smaller than video frame size
    imgPerson = cv2.imread('/Users/dominikwawak/Documents/FinalYear/Project/motionplanningStuff/USV-LIR-MotionPlanning-Vision/res/pp1.png', cv2.IMREAD_UNCHANGED)
    #imgPerson = cv2.resize(imgPerson,(50,50))
    imgUSV = cv2.imread('/Users/dominikwawak/Documents/FinalYear/Project/motionplanningStuff/USV-LIR-MotionPlanning-Vision/res/usv.png', cv2.IMREAD_UNCHANGED)
    imgUSV = cv2.resize(imgUSV,(250,450))

    # Get Image dimensions
    imgPerson_height, imgPerson_width, _ = imgPerson.shape
    imgUSV_height, imgUSV_width, _ = imgUSV.shape

    # Set initial position of images and used for change 
    xp=50
    yp=50
    deltaXUSV=50
    deltaYUSV=50

  

    while True:
        
        global path_finished


        
        success,img = video.read()
        success,img2 = video.read()

        if img is None:
            break
        if img is not None and img.size > 0:
            image = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)



        # array for valid circle points
        
        valid_circles=[]
        G = nx.Graph()
        valid_contours=[]
        startPoint=()
        endPoint=()


        if simultaion_mode: 
            # add image to frame
            #img[ yp:yp+imgPerson_height, xp:xp+imgPerson_width]=imgPerson
            img = cvzone.overlayPNG(img, imgPerson, [xp,yp])
            if usv_simulation:
                img = cvzone.overlayPNG(img, imgUSV, [(width-deltaXUSV)-imgUSV_width,(height-deltaYUSV)-imgUSV_height])
           
        
        


        #****************************************************************************************************
        #*************COLOR DETECTION_START************************************************************************
        
        lower = np.array([app.LH,app.LS,app.LV])
        upper = np.array([app.UH,app.US,app.UV])

        mask = cv2.inRange(image,lower,upper)
        mask = cv2.bitwise_not(mask)
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big_countours=[]
        usv_contour=None
        person_contour=None


        for cnt in contours:
            area=cv2.contourArea(cnt)
            if area>10000 and area<150000:
                #print(area)
                x,y,w,h = cv2.boundingRect(cnt)
                big_countours.append(cnt)
                cv2.rectangle(mask, (x-40,y-40),(x+w+40,y+h+40), (255,255,255),-1)

        # drawing circles        
        for i in pointGrid:
            if(mask[i[1],i[0]].sum()==0 and not ignore_water):
                valid_circles.append((i[0],i[1]))
                cv2.circle(img2, (i[0],i[1]), 10, (0,0,255), 2)
            elif ignore_water:
                valid_circles.append((i[0],i[1]))
                cv2.circle(img2, (i[0],i[1]), 10, (0,0,255), 2)

        #****************************************************************************************************
        #*************COLOR DETECTION_END************************************************************************
            


        #****************************************************************************************************
        #*************OBJECT RECOGNITION _ START************************************************************************


        if start_path_planning:

            if local_model:
                blob=cv2.dnn.blobFromImage(img,1/255,(640,640),[0,0,0],swapRB=True,crop=False)
                net.setInput(blob)
                detections = net.forward()[0]
                # predicts 25200 boxes or detections, each box has 8 entries
                #print(detections.shape)

                #cx cy w h conf class 8 class scores
                # class+ids, confidence, bounding box

                classes_ids = []
                confidences = []
                boxes = []
                rows = detections.shape[0]

                #scale
                frame_width, frame_height = img.shape[1], img.shape[0]
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
                    cv2.rectangle(img,(x,y),(x+width,y+height),(0,255,0),2)
                    cv2.putText(img,text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

                    if label=="person":
                        endPoint=(int(x+width/2),int(y+height/2))
                        cv2.circle(img2, endPoint, 10, (0,255,0), -1)

                    if label=="usv":
                        startPoint=(int(x+width/2),int(y+height/2))
                        cv2.circle(img2, startPoint, 10, (0,255,0), -1)

                    if startPoint != () and endPoint != ():
                            # Detect if boat is close to the person 
                            if(abs(startPoint[0]-endPoint[0])<=100 and abs(startPoint[1]-endPoint[1])<=100):
                                print("Boat is close to the person")
                                client.publish("test/res", "stop1")
                                path_finished=True





            else:

                prediction=model.predict(img, confidence=40, overlap=30).json()

                bounding_box=None
                for i in range(0,3,1): 
                    try:

                        # print(prediction['predictions'][i]['class'])


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
                        pass
                        # print("Nothing found")

                    if(bounding_box):
                        if(bounding_box['class']=="usv" or bounding_box['class']=="2" or bounding_box['class']=="4"  ):
                            # for cnt in big_countours:
                            #     if( cv2.pointPolygonTest(cnt,(bounding_box['x'],bounding_box['y']),False)==1):
                            #         usv_contour=cnt
                            #         break
                            # x,y,w,h = cv2.boundingRect(usv_contour)
                            
                            # startPoint=(int(x+w/2),int(y+h/2))
                                
                            startPoint=(int(x0+bounding_box['width']/2), int(y0+bounding_box['height'] / 2))
                            cv2.circle(img2, startPoint, 10, (0,255,0), -1)
                            
                                
                        if(bounding_box['class']=="person" or bounding_box['class']=="0"):
                            # for cnt in big_countours:
                            #     if( cv2.pointPolygonTest(cnt,(bounding_box['x'],bounding_box['y']),False)==1):
                            #         person_contour=cnt
                            #         break

                            # x,y,w,h = cv2.boundingRect(person_contour)  
                            # endPoint=(int(x+w/2),int(y+h/2))
                            endPoint=(int(x0+bounding_box['width']/2), int(y0+bounding_box['height'] / 2))
                            cv2.circle(img2, endPoint, 10, (0,255,0), -1)

                        if startPoint != () and endPoint != ():
                            # Detect if boat is close to the person 
                            if(abs(startPoint[0]-endPoint[0])<=500 and abs(startPoint[1]-endPoint[1])<=200):
                                print("Boat is close to the person")
                                client.publish("test/res", "stop1")
                                path_finished=True

                        


            
            #****************************************************************************************************
            #*************OBJECT RECOGNITION _END************************************************************************





    
            # boat_ready_msg!="")
            if(startPoint!=() and endPoint!=() and not path_finished) : # if both points are found -> crate the graph

                if(abs(startPoint[0]-startPoint_prev[0])<=101 and abs(endPoint[0]-endPoint_prev[0])<=101):
                    validPointCount+=1
                    print("validPointCount",validPointCount)
                    if(validPointCount==2):
                        point_drift=(startPoint[0]-startPoint_prev[0],startPoint[1]-startPoint_prev[1])
                        valid_circles_path_found=valid_circles



                        start_nearest = min(valid_circles, key=lambda x: sqrt((startPoint[0] - x[0])**2 + (startPoint[1] - x[1])**2))
                        nearest_dist =sqrt((startPoint[0] - start_nearest[0])**2 + (startPoint[1] - start_nearest[1])**2)

                        end_nearest = min(valid_circles, key=lambda x: sqrt((endPoint[0] - x[0])**2 + (endPoint[1] - x[1])**2))
                        nearest_dist_end=sqrt((endPoint[0] - end_nearest[0])**2 + (endPoint[1] - end_nearest[1])**2)

                        valid_circles.append(startPoint)
                        valid_circles.append(endPoint)
                        validPointCount=0

                        if not local_model:
                            if(usv_width_pixels==0):
                                usv_width_pixels=abs(start_point[0]-end_point[0])
                                print("USV WIDTH IN PIXELS",usv_width_pixels)
                                pixels_per_cm=usv_width_pixels/usv_width_cm
                                print("PIXELS PER CM",pixels_per_cm)
                    

                        # Create a graph
                        for point in valid_circles:
                            x, y = point
                            # if(x>= startPoint[0] and x<=endPoint[0]): # minimize the amount of points
                            G.add_node(point)

                    
                        
                        # Add edges between nodes with weights as Euclidean distances
                        for i in range(len(valid_circles)-1):
                            for j in range(i+1, len(valid_circles)):
                                
                                x1, y1 = valid_circles[i]
                                x2, y2 = valid_circles[j]
                                
                                distance = sqrt((x1 - x2)**2 + (y1 - y2)**2)
                                if distance <= 50:
                                    G.add_edge(valid_circles[i], valid_circles[j], weight=distance)

                        
                        
                        print("nearest_dist",nearest_dist,"nearest_dist_end",nearest_dist_end)
                        print("start_nearest",start_nearest,"end_nearest",end_nearest,"startPoint",startPoint,"endPoint",endPoint)


                        # Add edges between nodes outside and the box 

                        for circ in valid_circles:
                            distance_toStart = sqrt((startPoint[0] - circ[0])**2 + (startPoint[1] - circ[1])**2)
                            distance_to_end = sqrt((endPoint[0] - circ[0])**2 + (endPoint[1] - circ[1])**2)
                            if(nearest_dist+50>=distance_toStart):
                                G.add_edge(startPoint, circ, weight=distance_toStart)
                                cv2.circle(img, circ, 10, (0,255,0), -1)
                        
                            if (nearest_dist_end+50>=distance_to_end):
                                G.add_edge(endPoint, circ, weight=distance_to_end)
                                cv2.circle(img, circ, 10, (0,255,0), -1)
                        




                        

                        # G.add_edge(startPoint, start_nearest, weight=sqrt((start_point[0] - start_nearest[0])**2 + (start_point[1] - start_nearest[1])**2))

                        # end_nearest = min(valid_circles, key=lambda x: sqrt((end_point[0] - x[0])**2 + (end_point[1] - x[1])**2))
                        # G.add_edge(endPoint, end_nearest, weight=sqrt((end_point[0] - end_nearest[0])**2 + (end_point[1] - end_nearest[1])**2))
                
                
                        try:
                            
                            shortest_path = nx.dijkstra_path(G, startPoint, endPoint, weight='weight')
                            distance = nx.dijkstra_path_length(G, startPoint, endPoint, weight='weight')
                            directions=[]
                            distances=[]


                            print("Shortest PATH",shortest_path)
                            print("Shortest PATH DISTANCE",distance)
                            if not local_model:
                                print("Shortest PATH DISTANCE IN CM",distance/pixels_per_cm)

                            for i in range(len(shortest_path)):
                                if(i+1<len(shortest_path)):
                                    cv2.line(img2, (shortest_path[i]), (shortest_path[i+1]), (0, 255, 0), thickness=3, lineType=8)
                                    cv2.line(img, (shortest_path[i]), (shortest_path[i+1]), (0, 255, 0), thickness=3, lineType=8)
                                    distances.append(sqrt((shortest_path[i][0]-shortest_path[i+1][0])**2 + (shortest_path[i][1]-shortest_path[i+1][1])**2))
                                    # # Directions
                                    # if(shortest_path[i][0]<shortest_path[i+1][0] and shortest_path[i][1]<shortest_path[i+1][1]):
                                    #     print("Diagonal Down Right")
                                    #     directions.append("Diagonal Down Right")
                                    # elif(shortest_path[i][0]<shortest_path[i+1][0] and shortest_path[i][1]>shortest_path[i+1][1]):
                                    #     print("Diagonal Up Right")
                                    #     distances
                                    #     directions.append("Diagonal Up Right")
                                    # elif(shortest_path[i][0]>shortest_path[i+1][0] and shortest_path[i][1]<shortest_path[i+1][1]):
                                    #     print("Diagonal Down Left")
                                    #     directions.append("Diagonal Down Left")
                                    # elif(shortest_path[i][0]>shortest_path[i+1][0] and shortest_path[i][1]>shortest_path[i+1][1]):
                                    #     print("Diagonal Up Left")
                                    #     directions.append("Diagonal Up Left")
                                    if(shortest_path[i][0]<shortest_path[i+1][0] and shortest_path[i][1]==shortest_path[i+1][1]):
                                        print("Right")
                                        directions.append("east")
                                        #directions.append("north")
                                    elif(shortest_path[i][0]>shortest_path[i+1][0] and shortest_path[i][1]==shortest_path[i+1][1]):
                                        print("Left")
                                        directions.append("west")
                                        #directions.append("south")
                                    elif(shortest_path[i][0]==shortest_path[i+1][0] and shortest_path[i][1]<shortest_path[i+1][1]):
                                        print("Down")
                                        directions.append("south")
                                        #directions.append("east")
                                    elif(shortest_path[i][0]==shortest_path[i+1][0] and shortest_path[i][1]>shortest_path[i+1][1]):
                                        print("Up")
                                        directions.append("north")
                                        #directions.append("west")
                                    else:
                                        print("No Direction")   
                            path_found=True
                            print("Directions",directions)

                            if(len(directions)>0):
                                if(len(directions)==1):
                                    print("sending STOP")
                                    client.publish("test/res", "stop1")
                                else:
                                    print("sending driection",directions[0])
                                    client.publish("test/res", directions[0])

                                    if simultaion_mode and usv_simulation:
                                        if(directions[0]=="north"):
                                            #move boat image up 50 pixels in the image and rotate image
                                            deltaYUSV=deltaYUSV+50
                                        elif(directions[0]=="south"):
                                            #imgUSV = cv2.rotate(imgUSV, cv2.ROTATE_180)
                                            #imgUSV_height, imgUSV_width, _ = imgUSV.shape
                                            deltaYUSV=deltaYUSV-50
                                        elif(directions[0]=="east"):
                                            #imgUSV= cv2.rotate(imgUSV, cv2.ROTATE_90_CLOCKWISE)
                                            deltaXUSV=deltaXUSV-50
                                            imgUSV_height, imgUSV_width, _ = imgUSV.shape
                                        elif(directions[0]=="west"):
                                            #imgUSV= cv2.rotate(imgUSV, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                            deltaXUSV=deltaXUSV+50
                                            #imgUSV_height, imgUSV_width, _ = imgUSV.shape

                                        else:
                                            print("No Direction")

                                
                        

                        except Exception as e:
                            print("No Path",e)
                            pass
                            #print("no Path")
                        
                    startPoint_prev=startPoint
                    endPoint_prev=endPoint
                else:
                    startPoint_prev=startPoint
                    endPoint_prev=endPoint
                    validPointCount=0
                
        app.show_frame(mask,img,img2)
        
    
        cv2.waitKey(1)



class App:

    bigImage=0
    currentLH = 90
    currentLS = 150
    currentLV = 20
    currentUH = 138
    currentUS = 255
    currentUV = 255

    selectedModel=""

    def on_label_click1(event):
        App.bigImage=1
        # print("Label1 clicked!",App.bigImage)
        

    def on_label_click2(event):
        App.bigImage=2
        #print("Label2 clicked!")
        

    def on_label_click3(event):
        App.bigImage=3
        #print("Label3 clicked!")

    def update_slider(self, name, val):
        
        #print("{}: {}".format(name, val))
        if name == "LH":
            self.LH = int(val)
        if name == "LS":
            self.LS= int(val)
        if name == "LV":
            self.LV= int(val)
        if name == "UH":
            self.UH = int(val)
        if name == "US":
           self.US= int(val)
        if name == "UV":
           self.UV = int(val)
        #print(self.LH)
    
    def streamCameraClick(self):
        t1=threading.Thread(target=motion_recognitionThread,args=[1,app.selectedModel])
        t1.start()
    def streamFileClick(self):
        t1=threading.Thread(target=motion_recognitionThread,args=[2,app.selectedModel])
        t1.start()

    def streamUrlClick(self):
        t1=threading.Thread(target=motion_recognitionThread,args=[3,app.selectedModel])
        t1.start()

    def selectRobo(self, selected_option):
        print(f"Selected option: {selected_option}")
        app.selectedModel=selected_option
        # do something with selected_option here
    def toggle_start_path():
        global start_path_planning
        start_path_planning = not start_path_planning

    def toggle_ignore_water():
        global ignore_water
        ignore_water = not ignore_water
        
        
        
        

    def __init__(self, master):

        self.LH=0
        self.LS=0
        self.LV=0
        self.UH=0
        self.US=0
        self.UV=0

        self.master = master
        master.title("Video Stream")

      
        rightFrame=Frame(master)
        rightFrame.pack(side='right')

        topButtonsFrame = Frame(master)
        topButtonsFrame.pack(side='top')

        slidersFrame = Frame(master)
        slidersFrame.pack(side="top")

        mainVideoWindow = Frame(master)
        mainVideoWindow.pack(side='left')

        # sideWindow = Frame(master)
        # sideWindow.pack(pady=10)



        self.sliderLH = Scale(slidersFrame, from_=0, to=255, orient=VERTICAL,label="LH",command=lambda val: self.update_slider("LH", val))
        self.sliderLH.pack(side=LEFT)
        self.sliderLS = Scale(slidersFrame, from_=0, to=255, orient=VERTICAL,label="LS",command=lambda val: self.update_slider("LS", val))
        self.sliderLS.pack(side=LEFT)
        self.slidersLV= Scale(slidersFrame, from_=0, to=255, orient=VERTICAL,label="LV",command=lambda val: self.update_slider("LV", val))
        self.slidersLV.pack(side=LEFT)
        self.sliderUH = Scale(slidersFrame, from_=0, to=255, orient=VERTICAL,label="UH",command=lambda val: self.update_slider("UH", val))
        self.sliderUH.pack(side=LEFT)
        self.sliderUS = Scale(slidersFrame, from_=0, to=255, orient=VERTICAL,label="US",command=lambda val: self.update_slider("US", val))
        self.sliderUS.pack(side=LEFT)
        self.sliderUV = Scale(slidersFrame, from_=0, to=255, orient=VERTICAL,label="UV",command=lambda val: self.update_slider("UV", val))  
        self.sliderUV.pack(side=LEFT)



        self.sliderLH.set(90)
        self.sliderLS.set(150)
        self.slidersLV.set(20)
        self.sliderUH.set(138)
        self.sliderUS.set(255)
        self.sliderUV.set(255)


        # self.console=Text(slidersFrame, height=100, width=300)
        # self.console.pack(side=RIGHT)

        self.stream1_button = Button(topButtonsFrame, text="Start Camera Stream")
        self.stream1_button.pack(side=LEFT)
        self.stream1_button.bind("<Button-1>", App.streamCameraClick)

        self.stream2_button = Button(topButtonsFrame, text="Start Local file Stream")
        self.stream2_button.pack(side=LEFT)
        self.stream2_button.bind("<Button-1>", App.streamFileClick)
        

        self.stream3_button = Button(topButtonsFrame, text="Start Url Stream")
        self.stream3Text = Entry(topButtonsFrame, text="Enter Url",)
        self.stream3_button.bind("<Button-1>", App.streamUrlClick)
        self.stream3_button.pack(side=LEFT)
        self.stream3Text.pack(side=LEFT)


        options = ["paper", "realLife","search_and_rescue","LOCAL-paper-version1","LOCAL-real-version1"]
        selection=StringVar()
        selection.set("paper")
        self.selectedModel="paper"
        self.selectRoboflow=OptionMenu(topButtonsFrame, selection, *options,command=lambda selected_option=selection.get(): self.selectRobo(selected_option))
        self.selectRoboflow.pack(side=RIGHT)


        self.start_path_planning_checkbox = Checkbutton(topButtonsFrame, text="Start Path Planning", command=App.toggle_start_path)
        self.start_path_planning_checkbox.pack(side=RIGHT)

        self.ignore_water_checkbox = Checkbutton(topButtonsFrame, text="Ignore Water", command=App.toggle_ignore_water)
        self.ignore_water_checkbox.pack(side=RIGHT)



        self.label1 = Label(rightFrame)
        self.label1.pack(side=BOTTOM)
        self.label1.bind("<Button-1>", App.on_label_click1)
          
        self.label2 = Label(rightFrame)
        self.label2.pack(side=BOTTOM)
        self.label2.bind("<Button-1>", App.on_label_click2)

        self.label3 = Label(rightFrame)
        self.label3.pack(side=BOTTOM)
        self.label3.bind("<Button-1>", App.on_label_click3)

        self.label4 = Label(mainVideoWindow)
        self.label4.pack()

        

        

   

    def show_frame(self,img1,img2,img3):
        frame = img1
        frame2 = img2
        frame3 = img3

        if(App.bigImage==1):
            frame4 = img1
        elif(App.bigImage==2):
            frame4 = img2
        elif(App.bigImage==3):
            frame4 = img3
        else:
            frame4 = img2


        frame = cv2.resize(frame, (300, 200))
        frame2 = cv2.resize(frame2, (300, 200))
        frame3 = cv2.resize(frame3, (300, 200))
        frame4 = cv2.resize(frame4, (720, 480))


        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        cv2image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
        cv2image3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGBA)
        cv2image4 = cv2.cvtColor(frame4, cv2.COLOR_BGR2RGBA)

        img_1 = Image.fromarray(cv2image)
        img_2 = Image.fromarray(cv2image2)
        img_3 = Image.fromarray(cv2image3)
        img_4 = Image.fromarray(cv2image4)

        imgtk1 = ImageTk.PhotoImage(image=img_1)
        imgtk2 = ImageTk.PhotoImage(image=img_2)
        imgtk3 = ImageTk.PhotoImage(image=img_3)
        imgtk4 = ImageTk.PhotoImage(image=img_4)

       
        self.label1.imgtk = imgtk1
        self.label1.configure(image=imgtk1)
        
        
        self.label2.imgtk = imgtk2
        self.label2.configure(image=imgtk2)
      

        self.label3.imgtk = imgtk3
        self.label3.configure(image=imgtk3)

        self.label4.imgtk = imgtk4
        self.label4.configure(image=imgtk4)
    

    def exit_program(self):
        
        self.master.destroy()




root = Tk()
root.geometry("1000x1000")
app = App(root)
root.mainloop()
    
    





