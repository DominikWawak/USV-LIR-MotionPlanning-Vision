
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import pafy
import cv2
import numpy as np
import config
import networkx as nx
from math import sqrt
import time
import threading
from roboflow import Roboflow
import math 
import paho.mqtt.client as mqtt



ack_msg=""



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
    ack_msg=m_decode.split(":")[1]

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


#****************************************************************************************************
#*************MQTT Setup************************************************************************



def motion_recognitionThread(option,mode):

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


    if(option==1):
        video = cv2.VideoCapture(0)
    elif(option==2):
        fn= askopenfilename()
        print("user chose", fn)
        video = cv2.VideoCapture(fn)
    elif(option==3):
        url = app.stream3Text.get()
        video = pafy.new(url)
        best = video.getbest(preftype="mp4")
        video = cv2.VideoCapture(best.url)

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
    G = nx.Graph()




    # 
    # Loop
    # 
    while True:

        
        success,img = video.read()
        success,img2 = video.read()
        image = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)


        # array for valid circle points
        valid_circles=[]
        
        startPoint=()
        endPoint=()


        #****************************************************************************************************
        #*************COLOR DETECTION_START************************************************************************
        
        lower = np.array([app.LH,app.LS,app.LV])
        upper = np.array([app.UH,app.US,app.UV])

        mask = cv2.inRange(image,lower,upper)
        mask = cv2.bitwise_not(mask)
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        for cnt in contours:
            area=cv2.contourArea(cnt)
            if area>10000 and area<150000:
                #print(area)
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(mask, (x-20,y-20),(x+w+20,y+h+20), (255,255,255),-1)

        # drawing circles        
        for i in pointGrid:
            if(mask[i[1],i[0]].sum()==0):
                valid_circles.append((i[0],i[1]))
                cv2.circle(img2, (i[0],i[1]), 10, (0,0,255), 2)

        # if(not path_found):
        #     # drawing circles        
        #     for i in pointGrid:
        #         if(mask[i[1],i[0]].sum()==0):
        #             valid_circles.append((i[0],i[1]))
        #             cv2.circle(img2, (i[0],i[1]), 10, (0,0,255), 2)
        # else:
        #     for i in valid_circles_path_found:
        #         cv2.circle(img2, (i[0],i[1]), 10, (0,0,255), 2)


        #****************************************************************************************************
        #*************COLOR DETECTION_END************************************************************************
            


        #****************************************************************************************************
        #*************OBJECT RECOGNITION _ START************************************************************************


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
                if(bounding_box['class']=="usv"):
                    startPoint=(int(x1)+40, int(y1-bounding_box['height'] / 2))
                    cv2.circle(img2, startPoint, 10, (0,255,0), -1)
                    
                        
                if(bounding_box['class']=="boat"):
                    endPoint=(int(x0)-40, int(y0+bounding_box['height'] / 2))
                    cv2.circle(img2, endPoint, 10, (0,255,0), -1)


        
        #****************************************************************************************************
        #*************OBJECT RECOGNITION _END************************************************************************





 

        if(startPoint!=() and endPoint!=() and not path_found): # if both points are found -> crate the graph

            if(abs(startPoint[0]-startPoint_prev[0])<=101 and abs(endPoint[0]-endPoint_prev[0])<=101):
                validPointCount+=1
                print("validPointCount",validPointCount)
                if(validPointCount==4):
                    point_drift=(startPoint[0]-startPoint_prev[0],startPoint[1]-startPoint_prev[1])
                    valid_circles_path_found=valid_circles
                    valid_circles.append(startPoint)
                    valid_circles.append(endPoint)

                    if(usv_width_pixels==0):
                        usv_width_pixels=abs(start_point[0]-end_point[0])
                        print("USV WIDTH IN PIXELS",usv_width_pixels)
                        pixels_per_cm=usv_width_pixels/usv_width_cm
                        print("PIXELS PER CM",pixels_per_cm)
                

                    # Create a graph object
                    #print(len(valid_circles),len(valid_circles_prev))
                
                    #print(len(valid_circles),len(valid_circles_prev))
                    for point in valid_circles:
                        x, y = point
                        if(x>= startPoint[0] and x<=endPoint[0]): # minimize the amount of points
                            G.add_node(point)

                        # Add edges between nodes with weights as Euclidean distance
                        for i in range(len(valid_circles)):
                            for j in range(i+1, len(valid_circles)):
                                
                                x1, y1 = valid_circles[i]
                                x2, y2 = valid_circles[j]
                            
                                #if (abs(x1-x2)<=73 or abs(x1-x2)==0) and (abs(y1-y2)<=73 or abs(y1-y2)==0):
                                if (abs(x1-x2)<=50 or abs(x1-x2)==0) and (abs(y1-y2)<=50 or abs(y1-y2)==0):
                                    distance = sqrt((x1 - x2)**2 + (y1 - y2)**2)
                                    G.add_edge(valid_circles[i], valid_circles[j], weight=distance)

                # elif(startPoint and endPoint and (path_found==3)):
                #     valid_circles.append(startPoint)
                #     valid_circles.append(endPoint)

                #     print("Shortest path",shortest_path)

                #     for node in G.nodes():
                #             if node not in shortest_path:
                #                 G.remove_node(node)


                
                 
                    else:

                        try:
                            
                            shortest_path = nx.dijkstra_path(G, startPoint, endPoint, weight='weight')
                            distance = nx.dijkstra_path_length(G, startPoint, endPoint, weight='weight')
                            directions=[]


                            print("Shortest PATH",shortest_path)
                            print("Shortest PATH DISTANCE",distance)
                            print("Shortest PATH DISTANCE IN CM",distance/pixels_per_cm)

                            for i in range(len(shortest_path)):
                                if(i+1<len(shortest_path)):
                                    cv2.line(img2, (shortest_path[i]), (shortest_path[i+1]), (0, 255, 0), thickness=3, lineType=8)
                                    # Directions
                                    if(shortest_path[i][0]<shortest_path[i+1][0] and shortest_path[i][1]<shortest_path[i+1][1]):
                                        print("Diagonal Down Right")
                                        directions.append("Diagonal Down Right")
                                    elif(shortest_path[i][0]<shortest_path[i+1][0] and shortest_path[i][1]>shortest_path[i+1][1]):
                                        print("Diagonal Up Right")
                                        directions.append("Diagonal Up Right")
                                    elif(shortest_path[i][0]>shortest_path[i+1][0] and shortest_path[i][1]<shortest_path[i+1][1]):
                                        print("Diagonal Down Left")
                                        directions.append("Diagonal Down Left")
                                    elif(shortest_path[i][0]>shortest_path[i+1][0] and shortest_path[i][1]>shortest_path[i+1][1]):
                                        print("Diagonal Up Left")
                                        directions.append("Diagonal Up Left")
                                    elif(shortest_path[i][0]<shortest_path[i+1][0] and shortest_path[i][1]==shortest_path[i+1][1]):
                                        print("Right")
                                        directions.append("Right")
                                    elif(shortest_path[i][0]>shortest_path[i+1][0] and shortest_path[i][1]==shortest_path[i+1][1]):
                                        print("Left")
                                        directions.append("Left")
                                    elif(shortest_path[i][0]==shortest_path[i+1][0] and shortest_path[i][1]<shortest_path[i+1][1]):
                                        print("Down")
                                        directions.append("Down")
                                    elif(shortest_path[i][0]==shortest_path[i+1][0] and shortest_path[i][1]>shortest_path[i+1][1]):
                                        print("Up")
                                        directions.append("Up")
                                    else:
                                        print("No Direction") 

                            
                                        
                            path_found=True

                        except:
                            pass
                            #print("no Path")
                    
                startPoint_prev=startPoint
                endPoint_prev=endPoint
            else:
                startPoint_prev=startPoint
                endPoint_prev=endPoint
                validPointCount=0
        
        # if(path_found==True and startPoint!=() and endPoint!=0):
        if(path_found==True ):
            global ack_msg
            # shortestGraph = nx.Graph()

            # del shortest_path[0]
            # del shortest_path[-1]
        
            # shortest_path.append(startPoint)
            # shortest_path.append(endPoint)
            # print("Shortest PATH",shortest_path)

            # try:
            #     for i in range(len(shortest_path)):
            #         for j in range(i+1, len(shortest_path)):
            #             x1, y1 = shortest_path[i]
            #             x2, y2 = shortest_path[j]
            #             if (abs(x1-x2)<=50 or abs(x1-x2)==0) and (abs(y1-y2)<=50 or abs(y1-y2)==0):
            #                 distance = sqrt((x1 - x2)**2 + (y1 - y2)**2)
            #                 shortestGraph.add_edge(shortest_path[i], shortest_path[j], weight=distance)
            #     shortest_path = nx.dijkstra_path(shortestGraph, startPoint, endPoint, weight='weight')
                        
            # except:
            #     pass
            

            
            
            for i in range(len(shortest_path)):
                    if(i+1<len(shortest_path)):
                        # cv2.line(img2, (shortest_path[i][0]+ point_drift[0],shortest_path[i][1]+point_drift[1]), (shortest_path[i+1][0]+ point_drift[0],shortest_path[i+1][1]+point_drift[1]), (0, 255, 0), thickness=3, lineType=8)
                          cv2.line(img2, (shortest_path[i]), (shortest_path[i+1]), (0, 255, 0), thickness=3, lineType=8) 

            ack=0
            for i in range(2):
                if(len(directions)>0):
                    print("sengind driection",directions[i])
                    client.publish("test/res", '{"data":{directions[i]},"ispublic":false}')
                    while(ack==0):
                        #ack=recievedMessage
                        print("waiting for ack",ack_msg)
                        if(ack_msg[1:-1]=="ack"):
                            ack=1
                            ack_msg=""
                        time.sleep(3)
                    
                    print("ack recieved",ack)

                ack=0
            path_found=False                       


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


        options = ["paper", "realLife"]
        selection=StringVar()
        selection.set("paper")
        self.selectedModel="paper"
        self.selectRoboflow=OptionMenu(topButtonsFrame, selection, *options,command=lambda selected_option=selection.get(): self.selectRobo(selected_option))
        self.selectRoboflow.pack(side=RIGHT)



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
    
    





