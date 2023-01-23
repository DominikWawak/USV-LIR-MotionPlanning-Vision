import networkx as nx
from math import sqrt

# Create a graph object
G = nx.Graph()

# Add coordinate points as nodes
coordinate_points = [(25,125),(100,50),(150,50),(50,100),(100,100),(100,150),(150,150),(200,150),(200,200),(250,150),(300,100),
(350,100),(400,100),(400,150),(400,200),(450,100),(450,150),(450,200),(375,125)]
for point in coordinate_points:
    G.add_node(point)

# Add edges between nodes with weights as Euclidean distance
for i in range(len(coordinate_points)):
    for j in range(i+1, len(coordinate_points)):
        
        x1, y1 = coordinate_points[i]
        x2, y2 = coordinate_points[j]
        # distance = sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # G.add_edge(coordinate_points[i], coordinate_points[j], weight=distance)
        if (abs(x1-x2)<=75 or abs(x1-x2)==0) and (abs(y1-y2)<=75 or abs(y1-y2)==0):
            distance = sqrt((x1 - x2)**2 + (y1 - y2)**2)
            G.add_edge(coordinate_points[i], coordinate_points[j], weight=distance)


# Find the shortest path between start point and end point using Dijkstra's algorithm
start = (25,125)
end = (375,125)
# G.remove_edge(start, end)
shortest_path = nx.dijkstra_path(G, start, end, weight='weight')

print(shortest_path)
