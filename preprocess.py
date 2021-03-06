import cv2
import numpy as np
from matplotlib import pyplot as plt


# EDGE DECTECTION

image = cv2.imread('areas/edge.jpg',0)
edges = cv2.Canny(image,50,70)
blur = cv2.GaussianBlur(edges,(5,5),100)
cv2.imwrite('tmp/edge.jpg',blur)



#CORNER

filename = 'tmp/edge.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
#dst = cv2.dilate(dst,None)

cornerimg = cv2.convertScaleAbs(dst)

cornershow = cornerimg.copy()

# iterate over pixels to get corner positions
w, h = gray.shape
for y in range(0, h):
  for x in range (0, w):
    #harris = cv2.cv.Get2D( cv2.cv.fromarray(cornerimg), y, x)
    #if harris[0] > 10e-06:
    if cornerimg[x,y] > 164:
     # print("corner at ", x, y)
      cv2.circle( cornershow,  # dest
	          (y,x),       # pos
	          4,           # radius
	          (255,0,0)    # color
	          )
cv2.imwrite('tmp/corner.jpg',cornershow)


# CONTOUR

im = cv2.imread('tmp/corner.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

len(contours)
cnt = contours[0]
print(len(cnt))

for h,cnt in enumerate(contours):
    # Draw rectangles around and inside obstacles
    rect = cv2.minAreaRect(cnt)
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    #print box
    if box[0][0] > 100:
        cv2.drawContours(im,[box]*2,0,(255,255,255),-50)
        cv2.drawContours(im,[box]*2,0,(255,255,255),50)



#SHOW IMAGES

plt.subplot(221),plt.imshow(image,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(cornershow,cmap = 'gray')
plt.title('Corner Image'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(im,cmap = 'gray')
plt.title('Contour Image'), plt.xticks([]), plt.yticks([])

cv2.imwrite('tmp/contour.jpg',im)

resized_image = cv2.resize(im, (640, 480)) 
cv2.imwrite('areas/area2.jpg',resized_image)

plt.show()


