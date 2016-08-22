import cv2
import numpy
import os
from heapq import *
from gasp import *

cam = cv2.VideoCapture(0)
file = "saved.png"

# A* Algorithm 
def heuristic(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

def astar(array, start, goal):

    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []

    heappush(oheap, (fscore[start], start))
    
    while oheap:

        current = heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j            
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue
                
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
                
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))
                path = heappush(oheap, (fscore[neighbor], neighbor))
                
    return False


nmap = numpy.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,0]])
# End of A* algorithm 


# Draw Path 
def draw_path(frame):
    path =  astar(nmap, (0,0), (9,12))
    # path_list = map(list, path)
    path_arr = numpy.array(path)
    multi_arr = numpy.array([50])
    path_array = path_arr * multi_arr
    x = 10
    y = 10
    for x in range(0,13):
        cv2.rectangle(frame,(0,0),(50,50),(0,255,0),1)
# End of Draw Path 

# Draw Grid
def grid(gray):
    x1 = 0
    x2 = 700
    for k in range(0, 700, 20):
        y1 = k
        y2 = k
        cv2.line(gray,(x1,y1),(x2,y2),(255,0,0),1)  

    y1 = 0
    y2 =700
    for k in range(0, 700, 20):
        x1 = k
        x2 = k
        cv2.line(gray,(x1,y1),(x2,y2),(255,0,0),1)   
# End of Draw Grid

# Crop Image
def crop():
    img = cv2.imread(file)
    i = 0
    os.mkdir('dataset2')  # Create a folder
    os.chdir('dataset2')  # Change directory
    for v in range(0, 640, 20):
        for c in range (0, 480, 20):
            crop_img = img[c:20+c, v:20+v] # Crop from x, y, w, h 
            #cv2.imshow("cropped", crop_img)
            cv2.imwrite(str(i) + '.png', crop_img)
            i += 1

# End of Crop Image

def run():
    stop = False
    print astar(nmap, (0,0), (9,12))

    if cam.isOpened(): # try to get the first frame
        rval, frame = cam.read()
    else:
        rval = False
        print "Camera is not found"

    while rval or not stop:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        key = cv2.waitKey(20)
        #print key, type(key)
        grid(gray)
        draw_path(gray)

        #cv2.putText(gray, "Press ESC to close.", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
        cv2.imshow('Obstacle Detection', gray)

        if key & 0xFF in [ord('S'), ord('s')]: # Screenshot
            cv2.imwrite(file, gray)
            print "Screenshot taken!"
        elif key & 0xFF in [ord('C'), ord('c')]: # Cropped
            crop()
            print "Cropped"
        elif key & 0xFF in [ord('Q'), ord('q')]: # Exit
            print "Exiting"
            stop = True
            break


    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
