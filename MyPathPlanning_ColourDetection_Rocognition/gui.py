
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




def motion_recognitionThread(option,model):

    
    if(model==1):
        rf = Roboflow(api_key=config.apiKeyPaper)
        project = rf.workspace().project("usvlirpaper")
        model = project.version(1).model
    elif(model==2):
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

    # video = cv2.VideoCapture(0)
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

        
        lower = np.array([app.LH,app.LS,app.LV])
        upper = np.array([app.UH,app.US,app.UV])

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

        app.show_frame(mask,img,img2)
       
        # cv2.imshow("points",image)
        # cv2.imshow("maks",mask)
        # cv2.imshow("webcam",img)
        # cv2.imshow("webcamspots",img2)

        cv2.waitKey(1)


class App:

    bigImage=0
    currentLH = 90
    currentLS = 150
    currentLV = 20
    currentUH = 138
    currentUS = 255
    currentUV = 255



    def on_label_click1(event):
        App.bigImage=1
        print("Label1 clicked!",App.bigImage)
        

    def on_label_click2(event):
        App.bigImage=2
        print("Label2 clicked!")
        

    def on_label_click3(event):
        App.bigImage=3
        print("Label3 clicked!")

    def update_slider(self, name, val):
        
        print("{}: {}".format(name, val))
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
        print(self.LH)
    
    def streamCameraClick(self):
        t1=threading.Thread(target=motion_recognitionThread,args=[1,1])
        t1.start()
    def streamFileClick(self):
        t1=threading.Thread(target=motion_recognitionThread,args=[2,2])
        t1.start()

    def streamUrlClick(self):
        t1=threading.Thread(target=motion_recognitionThread,args=[3,2])
        t1.start()
        

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
        topButtonsFrame.pack(side='top',pady=15)

        mainVideoWindow = Frame(master)
        mainVideoWindow.pack(pady=41)

        slidersFrame = Frame(master)
        slidersFrame.pack(side=LEFT,pady=15)

        
        # LH=90
        # LS=150
        # LV=20
        # UH=138
        # US=255
        # UV=255

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


        self.console=Text(slidersFrame, height=100, width=300)
        self.console.pack(side=RIGHT)

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
    

        # self.label.configure(image=imgtk)
        # self.label.pack(side="right")
        # sself.master.after(10, self.show_frame)


    def exit_program(self):
        
        self.master.destroy()




root = Tk()
root.geometry("1000x1000")
app = App(root)
root.mainloop()
    
    





