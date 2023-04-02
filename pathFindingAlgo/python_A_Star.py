# import heapq

# def A_star(start, goal, grid):
#     # Create a priority queue to store the nodes to be expanded
#     queue = [(0, start)]
#     # Create a dictionary to store the cost of each node
#     cost = {start: 0}
#     # Create a dictionary to store the parents of each node
#     parents = {start: None}
#     # Create a set to store the visited nodes
#     visited = set()
    
#     while queue:
#         # Pop the node with the lowest cost from the queue
#         current = heapq.heappop(queue)[1]
        
#         # If the current node is the goal, return the path
#         if current == goal:
#             path = []
#             while current != start:
#                 path.append(current)
#                 current = parents[current]
#             path.append(start)
#             return path[::-1]
        
#         # If the current node has already been visited, skip it
#         if current in visited:
#             continue
#         visited.add(current)
        
#         # Get the neighbors of the current node
#         neighbors = get_neighbors(current, grid)
        
#         # Iterate over the neighbors
#         for neighbor in neighbors:
#             # Calculate the new cost to reach the neighbor
#             new_cost = cost[current] + 1
            
#             # If the new cost is less than the current cost, update it
#             if neighbor not in cost or new_cost < cost[neighbor]:
#                 cost[neighbor] = new_cost
#                 priority = new_cost + heuristic(neighbor, goal)
#                 parents[neighbor] = current
#                 heapq.heappush(queue, (priority, neighbor))
                
#     # If the goal is not reached, return None
#     return None

# def get_neighbors(node, grid):
#     x, y = node
#     neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
#     neighbors = [n for n in neighbors if is_valid(n, grid)]
#     return neighbors

# def is_valid(node, grid):
#     x, y = node
#     if x < 0 or x >= len(grid) or y < 0 or y >= len(grid[0]):
#         return False
#     if grid[x][y] == "wall":
#         return False
#     return True

# def heuristic(node, goal):
#     x1, y1 = node
#     x2, y2 = goal
#     return abs(x1 - x2) + abs(y1 - y2)

# # Example usage
# grid = [[0, 0, 0, 0],
#         [0, "wall", 0, 0],
#         [0, "wall", 0, 0],
#         [0, 0, 0, 0]]
# start = (0, 0)
# goal = (3, 3)
# path = A_star(start, goal, grid)
# print(path)




# # Define a list of predefined points
# points = [(100, 200), (300, 400), (500, 600), (700, 800)]

# # Create a list to store the open and closed sets
# open_set = []
# closed_set = []

# # Define a dictionary to store the came_from information for each point
# came_from = {}

# # Define a function to calculate the distance between two points
# def distance(point1, point2):
#     return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

# # Define a function to find the shortest path using A* algorithm
# def a_star(start, end, open_set, closed_set):
#     # While there are still points in the open set
#     while open_set:
#         # Find the point in the open set with the lowest f score (f = g + h)
#         current = None
#         current_f = None
#         for point in open_set:
#             g = distance(point, start)
#             h = distance(point, end)
#             f = g + h
#             if current_f is None or f < current_f:
#                 current_f = f
#                 current = point

#         # If the current point is the end point, return the path
#         if current == end:
#             path = [end]
#             while path[-1] != start:
#                 path.append(came_from[path[-1]])
#             return path[::-1]

#         # Remove the current point from the open set and add it to the closed set
#         open_set.remove(current)
#         closed_set.append(current)

#         # Check the neighboring points
#         for neighbor in points:
#             if neighbor in closed_set:
#                 continue
#             if neighbor not in open_set:
#                 open_set.append(neighbor)
#                 came_from[neighbor] = current
#     # If there are no more points in the open set, return None (no path found)
#     return None

# # Find the shortest path using A*
# for i in range(len(points) - 1):
#     start = points[i]
#     end = points[i + 1]
#     path = a_star(start, end, open_set, closed_set)
#     if path is not None:
#         print("Shortest path from", start, "to", end, ":", path)
#     else:
#         print("No path found from", start, "to", end)



points=[(0,1), (2,2), (5,5),(0,9),(7, 8),(9,9)]


openList=[]
closedList=[]

startPoint=(0,0)
endPoint=(9,9)
openList.append(startPoint)

# # Define a dictionary to store the came_from information for each point
came_from = {}

# euclydiean distance
def distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 10.5

def a_star(startPoint, endPoint, openList, closedList):
    while openList:
        # Find the point in the open set with the lowest f score (f = g + h)
            current = None
            current_f = None
            for point in openList:
                g = distance(point, startPoint)
                h = distance(point, endPoint)
                f = g + h
                if current_f is None or f < current_f:
                    current_f = f
                    current = point

            # If the current point is the end point, return the path
            if current == endPoint:
                path = [endPoint]
                while path[-1] != startPoint:
                    path.append(came_from[path[-1]])
                return path[::-1]
            
            # Remove the current point from the open set and add it to the closed set
            openList.remove(current)
            closedList.append(current)
            # Check the neighboring points
            for neighbor in points:
                if neighbor in closedList:
                    continue
                if neighbor not in openList:
                    openList.append(neighbor)
                    came_from[neighbor] = current
        # If there are no more points in the open set, return None (no path found)
    return None


# path = a_star(startPoint, endPoint, openList, closedList)
# if path is not None:
#         print("Shortest path from", startPoint, "to", endPoint, ":", path)
# else:
#         print("No path found from", startPoint, "to", endPoint)

# Find the shortest path using A*
# for i in range(len(points) - 1):
#     start = points[i]
#     end = points[i + 1]
#     openList=[]
#     closedList=[]
#     openList.append(start)
#     path = a_star(start, end, openList, closedList)
#     if path is not None:
#         print("Shortest path from", start, "to", end, ":", path)
#     else:
#         print("No path found from", start, "to", end)
path = a_star(startPoint, endPoint, openList, closedList)
if path is not None:
        print("Shortest path from", startPoint, "to", endPoint, ":", path)
else:
        print("No path found from", startPoint, "to", endPoint)






