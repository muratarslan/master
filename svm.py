import os
import sys
import fnmatch
import getopt
import cv2
import numpy as np
import numpy
import datetime
from sklearn import svm
from heapq import *
from gasp import *
from matplotlib import pyplot as plt
from math import degrees, atan2
from naoqi import ALProxy


number_of_bins = 64
positive = 'dataset/positive'
negative = 'dataset/negative'
ds = 'dataset'
data = []
angle = 0


# Get all 'png' images from 'negative' and 'positive' folder
def getImages():
    imageFiles = []
    for i in range(2):
        if i == 0:
            path = positive
        elif i == 1:
            path = negative
        for j in sorted(os.listdir(path)):
            if fnmatch.fnmatch(j, '*.png'):
                imageFiles.append(j)
        i += 1
    return imageFiles


# Returns histogram result
def getHistogram(imageFiles):
    image = cv2.imread(imageFiles)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray],[0],None,[number_of_bins],[0,number_of_bins])
    transp = histogram.transpose()
    return transp.astype(np.float64)


# Gets all pictures' histograms
def getHistograms():
    images = getImages()
    histogramMap = {}
    for i in images:
        im = positive +'/'+ i
        histogramMap[im] = getHistogram(im)
    for i in images:
        im = negative +'/'+ i
        histogramMap[im] = getHistogram(im)
    return histogramMap.values()


# Sets the values positive 1 negative 0 for svm values
def getValues():
    values = []
    for i in range(2):
        if i == 0:
            path = positive
            j = 0
        elif i == 1:
            path = negative
            j = 1
        for i in sorted(os.listdir(path)):
            values.append((j,))
    return values


# Splits the matrix in desired format
def split(mtx,num):
    matrix = np.array(mtx)
    matrix_splitted = np.array(np.split(matrix, num))
    return np.flipud(matrix_splitted)
    
    
# SVM learn and classify
def train():
    now = datetime.datetime.now()
    array = []
    trainData = map(lambda x: x[0], getHistograms())
    value = getValues()

    classify = svm.SVC(kernel='linear')
    classify.fit(trainData, value)


    for j in range(768):
        predict = getHistogram(ds +'/'+ str(j)+'.png')
        result = classify.predict(predict)

        if result == [1]:
            i = 0
        else:
            i = 1
        array.append(i)
    array = split(array,32).T
    #array = np.flipud(array)
    array = np.fliplr(array)

    print " "
    print "Prediction Time : " + str(datetime.datetime.now() - now)
    print " "
    print "##################################################"
    print "                     Area                        "
    print "##################################################"
    print " "
    print array

    #array[0,0] = 3
    return array
    
    

# A* Algorithm 
def heuristic(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

def path():
    
    array = train()
    start = (24,1)
    goal  = (2,25)
    
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

    closeSet = set()
    cameFrom = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []

    heappush(oheap, (fscore[start], start))
    
    while oheap:

        current = heappop(oheap)[1]

        if current == goal:
            global data
            while current in cameFrom:
                data.append(current)
                current = cameFrom[current]


            ## Draws path
	    img = cv2.imread('areas/edge.jpg',0)

            for i in range(0,len(data)) :
              	  x = int(data[i:][0][0])  # takes first element of first list
              	  y = int(data[i:][0][1])  # takes second element of first list
		  j = i + 1
		  if j>len(data)-1:
			j = len(data)-1     # should be -1 coz no more list
		  z = int(data[j:][0][0]) # takes first element of second list
		  t = int(data[j:][0][1]) # takes second element of second list
                  array[x,y] = 4   # puts 4 in array
		  cv2.circle(img,(y*20,x*20), 2, (255,0,0), -1)  # circle path
		  cv2.line(img,(y*20,x*20),(t*20,z*20),(5,5,2),5)  # line path

                #print x
                #print y
            print " "
            print "##################################################"
            print "                Area  with PATH                   "
            print "##################################################"
            print " "
            print array
	    print data[::-1]

	    cv2.imwrite('areas/area-path.png',img)

            # SHOW PICTURES
	    org = cv2.imread('areas/edge.jpg',0)
	    path = cv2.imread('areas/area-path.png',0)
	    plt.subplot(221),plt.imshow(org,cmap = 'gray')
	    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	    plt.subplot(222),plt.imshow(path,cmap = 'gray')
	    plt.title('Path Image'), plt.xticks([]), plt.yticks([])
	    plt.show()

            return data

        closeSet.add(current)
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
                
            if neighbor in closeSet and tentative_g_score >= gscore.get(neighbor, 0):
                continue
                
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                cameFrom[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))
                path = heappush(oheap, (fscore[neighbor], neighbor))

    return False
    
# End of A* algorithm 


# Bearing Algorithm
def gb(x, y, center_x, center_y):
	global angle
	angle = degrees(atan2(y - center_y, x - center_x))
	bearing1 = (angle + 360) % 360
	bearing2 = (90 - angle) % 360  
	#print "gb: x=%2d y=%2d angle=%6.1f bearing1=%5.1f bearing2=%5.1f" % (x, y, angle, bearing1, bearing2)
	#print angle
	return angle	

def moveToPoint():
	checkpoint = []
	point_list = []
	way_point = []
	angle_point = []
	coordinates = []
	cm = 0.07
	for i in range(len(data)-1):
		gb(data[i][0],data[i][1],data[i+1][0],data[i+1][1])
		first_point = (data[i][0],data[i][1])
		second_point = (data[i+1][0],data[i+1][1])
		lst = (angle, first_point, second_point)
		point_list.append(lst)

	way_point.append(point_list[0][1])

	for i in range(len(point_list)-1):
		if point_list[i][0] != point_list[i+1][0]:
			way_point.append(point_list[i+1][1])

	way_point.append(point_list[len(point_list)-1][1])
	way_point = way_point[::-1]

	print point_list
	print way_point

	for i in range(len(way_point)-1):
		gb(way_point[i][0],way_point[i][1],way_point[i+1][0],way_point[i+1][1])
		first_point = (way_point[i][0],way_point[i][1])
		second_point = (way_point[i+1][0],way_point[i+1][1])
		lst = (angle,second_point)
		if i == 0:
			angle_point.append(data[-1])
		angle_point.append(lst)
		if i == 0:
			coordinates.append(tuple(numpy.subtract(angle_point[0], angle_point[1][1])))
		if i > 0:
			coordinates.append(tuple(numpy.subtract(angle_point[i][1], angle_point[i+1][1])))
	print coordinates

	motion = ALProxy("ALMotion", "192.168.10.13", 9559)
	tts = ALProxy("ALTextToSpeech", "192.168.10.13", 9559)
	postureProxy = ALProxy("ALRobotPosture", "192.168.10.13", 9559)
	motion.moveInit()
	motion.setStiffnesses("Body", 1.0)
	motion.setMoveArmsEnabled(True, True)
	motion.setMotionConfig([["ENABLE_FOOT_CONTACT_PROTECTION", False]])
        motion.setExternalCollisionProtectionEnabled( "All", False )
        print "Protection Off"
        tts.say("Hummm, it seams easy!")

	for i in range(len(coordinates)):
		print "Walking"
		print coordinates[i][0]*cm, coordinates[i][1]*cm
		if coordinates[i][0] > 0 and coordinates[i][1] < 0:
			motion.moveTo(0, 0, -0.785398,
       			 [ ["MaxStepFrequency", 0.5],  # low frequency
          		   ["TorsoWy", 0.1] ])         # torso bend 0.1 
			motion.waitUntilMoveIsFinished()
			motion.moveTo(coordinates[i][0] * cm, 0, 0,
       			 [ ["MaxStepFrequency", 0.5],  # low frequency
          		   ["TorsoWy", 0.1] ])         # torso bend 0.1 rad in front)
			motion.waitUntilMoveIsFinished()
			motion.moveTo(0, 0, 0.785398,
       			 [ ["MaxStepFrequency", 0.5],  # low frequency
          		   ["TorsoWy", 0.1] ])         # torso bend 0.1 
			motion.waitUntilMoveIsFinished()

		elif coordinates[i][0] > 0 and coordinates[i][1] == 0:
			motion.moveTo(coordinates[i][0] * cm, 0, 0,
       			 [ ["MaxStepFrequency", 0.5],  # low frequency
          		   ["TorsoWy", 0.1] ])         # torso bend 0.1 rad in front))
			motion.waitUntilMoveIsFinished()

		elif coordinates[i][0] > 0 and coordinates[i][1] > 0:
			motion.moveTo(0, 0, 0.785398,
       			 [ ["MaxStepFrequency", 0.5],  # low frequency
          		   ["TorsoWy", 0.1] ])         # torso bend 0.1
			motion.waitUntilMoveIsFinished()
			motion.moveTo(coordinates[i][0] * cm, 0, 0,
       			 [ ["MaxStepFrequency", 0.5],  # low frequency
          		   ["TorsoWy", 0.1] ])         # torso bend 0.1 rad in front))
			motion.waitUntilMoveIsFinished()
			motion.moveTo(0, 0, -0.785398,
       			 [ ["MaxStepFrequency", 0.5],  # low frequency
          		   ["TorsoWy", 0.1] ])         # torso bend 0.1 
			motion.waitUntilMoveIsFinished()

		elif coordinates[i][0] == 0 and coordinates[i][1] < 0:
			motion.moveTo(0, 0, -1.5708,
       			 [ ["MaxStepFrequency", 0.5],  # low frequency
          		   ["TorsoWy", 0.1] ])         # torso bend 0.1
			motion.waitUntilMoveIsFinished()
			motion.moveTo(abs(coordinates[i][1]) * cm, 0, 0,
       			 [ ["MaxStepFrequency", 0.5],  # low frequency
          		   ["TorsoWy", 0.1] ])         # torso bend 0.1 rad in front))
			motion.waitUntilMoveIsFinished()
			motion.moveTo(0, 0, 1.5708,
       			 [ ["MaxStepFrequency", 0.5],  # low frequency
          		   ["TorsoWy", 0.1] ])         # torso bend 0.1 
			motion.waitUntilMoveIsFinished()

		elif coordinates[i][0] < 0 and coordinates[i][1] == 0:
			motion.moveTo(0, 0, 1.5708,
       			 [ ["MaxStepFrequency", 0.5],  # low frequency
          		   ["TorsoWy", 0.1] ])         # torso bend 0.1
			motion.waitUntilMoveIsFinished()
			motion.moveTo(abs(coordinates[i][0]) * cm, 0, 0,
       			 [ ["MaxStepFrequency", 0.5],  # low frequency
          		   ["TorsoWy", 0.1] ])         # torso bend 0.1 rad in front))
			motion.waitUntilMoveIsFinished()
			motion.moveTo(0, 0, -1.5708,
       			 [ ["MaxStepFrequency", 0.5],  # low frequency
          		   ["TorsoWy", 0.1] ])         # torso bend 0.1 
			motion.waitUntilMoveIsFinished()
	motion.moveTo(0.6,0,0,
	[ ["MaxStepFrequency", 0.5],  # low frequency
	  ["TorsoWy", 0.1] ])         # torso bend 0.1 )
	postureProxy.goToPosture("Sit", 1.0)
	tts.say("Finish")
	print "Finish"
		
	
