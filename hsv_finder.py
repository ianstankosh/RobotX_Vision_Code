import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/Users/ianstankosh/Desktop/HSV_Colors/red_upper.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# xbins, ybins = np.histogram2d(h.ravel(),s.ravel(),[180,256],[[0,180],[0,256]])

# hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

# plt.imshow(hist, interpolation='nearest')
# plt.show()

histr = cv2.calcHist([hsv], [0], None, [180], [0, 179])
plt.plot(histr, color='r')
plt.xlim([0, 180])

histr = cv2.calcHist([hsv], [1], None, [256], [0, 255])
plt.plot(histr, color='g')
plt.xlim([0, 256])

histr = cv2.calcHist([hsv], [2], None, [256], [0, 255])
plt.plot(histr, color='b')
plt.xlim([0, 256])

plt.show()
