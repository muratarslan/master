import cv2
import numpy as np

# Load image
im = cv2.imread('image.png')
img = cv2.imread('image.png')

# Kernel configuration
kernel = np.ones((35,35),np.uint8)


# Convert BGR to HSV
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)


# define range of white color in HSV
lower_blue = np.array([0,0,168])
upper_blue = np.array([179,255,255])


# Threshold the HSV image to get only green colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)


# Closing 
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


# Blur
blur = cv2.GaussianBlur(closing,(5,5),10)


# Getting contours
ret, thresh = cv2.threshold(blur, 127, 255,0)
contours,hierarchy = cv2.findContours(thresh,2,1)
cnt = contours[0]


# Get coordinates of a rectangle around the contour
leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])

print leftmost
print topmost
print bottommost
print rightmost


# Drawing lines and circles
hull = cv2.convexHull(cnt,returnPoints = False)
defects = cv2.convexityDefects(cnt,hull)

for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(im,start,end,[0,255,0],2)
    cv2.circle(im,far,5,[0,0,255],-1)


# Corner coordinates
#coordinates = [(topmost[0],topmost[1]), (bottommost[0], topmost[1]),(rightmost[0], rightmost[1]), (leftmost[0],leftmost[1])]
coordinates = [(155,348),(471,352),(569,402),(58,394)]


# Put coordinates in array
pts = np.array(coordinates, dtype = "float32")
print pts
(tl, tr, br, bl) = pts


# compute the width of the new image, which will be the
# maximum distance between bottom-right and bottom-left
# x-coordiates or the top-right and top-left x-coordinates
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))


# compute the height of the new image, which will be the
# maximum distance between the top-right and bottom-right
# y-coordinates or the top-left and bottom-left y-coordinates
heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))


# now that we have the dimensions of the new image, construct
# the set of destination points to obtain a "birds eye view",
# (i.e. top-down view) of the image, again specifying points
# in the top-left, top-right, bottom-right, and bottom-left
# order
dst = np.array([
	[0, 0],
	[maxWidth - 1, 0],
	[maxWidth - 1, maxHeight - 1],
	[0, maxHeight - 1]], dtype = "float32")


# compute the perspective transform matrix and then apply it
M = cv2.getPerspectiveTransform(pts, dst)
warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

resize = cv2.resize(warped, (640,480))

# show the original and warped images
cv2.imshow("HSV", hsv)
cv2.imshow("Original", img)
cv2.imshow("Trapezoid found", im)
cv2.imshow("Warped", resize)
#cv2.imwrite("areas/edge.jpg",resize)
cv2.waitKey(0)
cv2.destroyAllWindows()
