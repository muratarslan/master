# Image Library
import Image
import cv
import cv2
import os
import glob
from naoqi import ALProxy
import numpy as np
import config as c
from sys import executable
from subprocess import Popen



image     = "areas/area.jpg"
directory = c.DATASET_DIRECTORY
negative = c.NEGATIVE
positive = c.POSITIVE


def openCamera():
	# Live Camera
	Popen([executable, 'video.py'])

def perspective():
	# Perspective Correction
	Popen([executable, 'perspective.py'])

def process():
	# Preprocessing
	Popen([executable, 'preprocess.py'])

def ball():
	# Preprocessing
	Popen([executable, 'ball.py'])

def takePhoto():

  # Setting up proxy
  cameraProxy = ALProxy("ALVideoDevice", c.IP, c.PORT)

  # Camera ID
  cameraProxy.kCameraSelectID = c.CAMERAID

  # Camera Parameters
  cameraProxy.setParam(cameraProxy.kCameraSelectID,c.CAMERAID)

  # Subscribe Camera Proxy
  videoClient = cameraProxy.subscribe("python_client", c.RESOLUTION, c.COLORSPACE, 5)

  # image[6] contains ASCII
  naoImage = cameraProxy.getImageRemote(videoClient)

  # Unsubscribe Camera Proxy
  cameraProxy.unsubscribe(videoClient)

  # Image Size and Pixel Array
  imageWidth  = naoImage[0]
  imageHeight = naoImage[1]
  imageArray  = naoImage[6]

  # Create an Image
  im = Image.frombytes("RGB", (imageWidth, imageHeight), imageArray)
  img = np.array(im)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #ret,thresh = cv2.threshold(graY144   ,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

  # Save image.
  cv2.imwrite('image.png',img)

  # Show image
  #cv2.imshow('Image',thresh)
  print "Photo Taken..."

def cropImage():
    img = cv2.imread("tmp/contour.jpg")
    i = 0
    if os.path.isdir(directory) == False:
        os.mkdir(directory)  # Create a folder
        os.chdir(directory)  # Change directory
    else:
        os.chdir(directory)  # Change directory
    
    os.mkdir("negative")
    os.mkdir("positive")

    for v in range(0, 640, 20):
        for c in range (0, 480, 20):
            crop_img = img[c:20+c, v:20+v] # Crop from x, y, w, h 
            #cv2.imshow("cropped", crop_img)
            cv2.imwrite(str(i) + '.png', crop_img)
            i += 1
    print "Image Cropped and dataset created..."


def nameChanger():
    global positive
    global negative
    os.chdir(negative)
    for i, f in enumerate(glob.glob('*.png')):
    	print "%s -> %s.png" % (f, i)
	os.rename(f, "%s.png" % i)

    os.chdir(positive)
    for i, f in enumerate(glob.glob('*.png')):
	print "%s -> %s.png" % (f, i)
	os.rename(f, "%s.png" % i)



def drawGrid():
    img = cv2.imread(image)
    x1 = 0
    x2 = 700
    for k in range(0, 700, 20):
        y1 = k
        y2 = k
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),1)  

    y1 = 0
    y2 =700
    for k in range(0, 700, 20):
        x1 = k
        x2 = k
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),1) 

    cv2.imwrite('image.png', img)
        

if __name__ == '__main__':
  showImage()
  

