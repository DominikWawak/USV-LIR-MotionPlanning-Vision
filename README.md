# USV LIR 2.0 - Path Planning for Maritime Search and Rescue applications


<img src="MyPathPlanning_ColourDetection_Rocognition/res/LIR.png" alt="LIR" />  

# Project Overview


<!-- ### USV LIR 2.0 is an unmanned surface vehicle (USV) codenamed LIR after the Irish folklore legend the Childeren of Lir. It is designed as an open research platform allowing citizen research and pimary and secondary education level research. This platform is designed with many use cases in mind, including search and rescue, environmental monitoring, and more. This project is being used as a showcase of third level engineering to primary and secondary school students throught the means of running a mission design competition. This project interested marine biologists from Catholic University of valencia and IMEDMAR Valencia Spain and they are excited for our second goal that is is a sucessful mission in Valencia harbour, Spain, where the USV is be used to autonomously perofrm one of the weekly manrine research data collecting tasks of collecting water temperature, salinity, and oxygen levels at different depths. 

### The unique thing about this project is the layers of technology located under the hood. There is the master controller called Pixhawk 3, which is a flight controller used on many autonomous drone project that controls the boat with the help of GPS. It connects to a laptop through telementary communication and a RadioLink controller is used as a master controller for manual control. QGround conrol and Mission Plannert are the two mission planing software tools used to configure the boat and make it follow waypoints. The big thing on this boat is the microbit interface. Microbits are small microprocessors that are easily programmed with virtual puzzle blocks. This feature is a unique selling point of the project and it is waht enables this to be an open research platform. This infrastructure is used to abstract the complexity, physicas and math of the boat and allow anyone with little to no computer science or engineering experience to start developing missions for the boat. 

### My focus of this project is maritime search and rescue. I am using the USV LIR 2.0 to detect people with risk of drowning in the water and perform path planning to navigate to the person. The object detection is achieved using object detecion training models with tools like tensorflow and the roboflow API and the path planning is achieved using python and the fusion of image processing methods. I also developed a simple GUI to allow the user to select the video stream they want to use, and to select the model they want to use for object detection. The GUI also has image processing functionality to a certain extent allowinf for realtime adjustements to be made to improve the path planning process. I hope that my project will be useful to others who are interested in developing similar projects and make it easier for them to get started having the basic set up at hand. -->

### The USV LIR 2.0 is an unmanned surface vehicle designed to serve as an open research platform. It is named after the Irish legend Children of Lir. This project showcases engineering to school students through a mission design competition and is being used for marine research by biologists from Catholic University of Valencia. The unique aspect of this project is the technology underneath. The master controller, Pixhawk 3, controls the boat with the help of GPS and is connected to a laptop through telemetry communication, while QGround Control is the mission planning software tool used to configure the boat. The microbit interface, a small microprocessor programmed with virtual puzzle blocks, is a standout feature that makes this platform accessible to anyone, regardless of their technical background. My primary focus is maritime search and rescue, using object detection training models and image processing methods to detect and rescue people in risk of drowning. I developed GUI to provide ease of use for future use. The path planning for the boat is done with common motion planning algorithms like Dijkstra and the communication is done through MQTT.


# Technologies 
### Python, tensorflow, Pixhawk, Microbit, MQTT, AWS, Roboflow, OpenCV, QGroundControl

# Repository Overview
### In this repository you will find two folders: MyPathPlanning folder and TFODCourseYT

<br>
<br>

## MyPathPlanning folder
#### The `MyPathPlanning` folder contains all of the code used in my project, including color detection, object detection using the roboflow API, and motion planning. I've made an effort to keep the code simple and easy to follow, so that others can reproduce the steps.

<br>
<br>

## TFODCourseYT folder
#### The `TFODCourseYT` folder contains materials from Nicholas Renotte's YouTube course on object detection and recognition. Nicholas is a well-known YouTuber in the field, and his course helped me to learn about training models from scratch, engineering them, and evaluating their performance. In this folder, you'll find some of my scripts and the models I've trained on my local machine. 

#### If you're interested in learning more about object detection and recognition, I highly recommend checking out Nicholas's course [here](Link to course).