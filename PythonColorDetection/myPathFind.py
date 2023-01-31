import math


coordinate_points = [(100,50),(150,50),(50,100),(100,100),(100,150),(150,150),(200,150),(200,200),(250,150),(300,100),
(350,100),(400,100),(400,150),(400,200),(450,100),(450,150),(450,200),(375,125)]

start=(25,125)

end=(375,125)

visited=[]

current=start

current_d=math.inf

end=False

def distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


while(not end):
    # pick shortest distance
    for point in coordinate_points:
        if distance(current,point)< current_d:
            current=point
            current_d=distance(start,point)
            visited.append(point)
            coordinate_points.remove(point)
        if current== end:
            end=True
    
    print(current)

