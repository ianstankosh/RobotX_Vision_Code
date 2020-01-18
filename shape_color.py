# import the necessary packages
import numpy as np
import cv2
import HSV_Processing


# load the image
#image = cv2.imread('frame' + str(HSV_Processing.count) + '.jpg')

# find all the 'red' shapes in the image
lower = np.array([0, 100, 100])
upper = np.array([10, 255, 255])
shapeMask = cv2.inRange(HSV_Processing.hsv, lower, upper)

# find the contours in the mask
(cnts, _) = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print "%d red shapes" % (len(cnts))
cv2.imshow("Mask", shapeMask)

# loop over the contours
for c in cnts:
    # draw the contour and show it
    cv2.drawContours(HSV_Processing.hsv, [c], -1, (0, 255, 0), 2)
    cv2.imshow("HSV", HSV_Processing.hsv)
    #cv2.waitKey(0)