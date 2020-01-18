import numpy as np
import cv2
import math

# Structuring element for morphology
kernel = np.ones((12,12),np.uint8) # 12by12  square

# VideoCapture using the only available camera
cap = cv2.VideoCapture(0)

# Will wait until black is seen (Code Reset) and used the location to only focus on panel while detecting the code
focused = False
# Hold set of seen colors
res = []
# define range of black in HSV
black_upper_hsv = np.array([179, 60, 95], dtype = "uint8")
black_lower_hsv = np.array([0, 0, 0], dtype = "uint8")



# Set up the detector with default parameters.
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
# Filter by Area.
params.filterByArea = True
params.minArea = 100
# Create detector
detector = cv2.SimpleBlobDetector(params)

# define the list of boundaries
boundaries = [
    (['R'], [17, 15, 100], [50, 56, 200]),      # R
    (['B'], [110, 100, 140], [160, 185, 255]),  # B
    (['Y'], [25, 146, 190], [62, 174, 250]),    # Y
    (['G'], [103, 86, 65], [145, 133, 128])     # G
]

imgFile = cv2.imread('/Users/ianstankosh/Desktop/HeadShot.jpg')
cv2.imshow('Original Black', imgFile)

hsv = cv2.cvtColor(imgFile, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV_BLK', hsv)

# Threshold the HSV image to get only input colors
blk = cv2.inRange(hsv, black_lower_hsv, black_upper_hsv)
#blk = cv2.bitwise_not(blk)
cv2.imshow('blk', blk)
closing = cv2.morphologyEx(blk, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Closing_blk', closing)

# Detect blobs.
keypoints = detector.detect(closing)

#Draw detected blobs as red circles.
#cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(closing, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Panel', im_with_keypoints)

panel = closing


imgFile2 = cv2.imread('BluePic.jpg')
cv2.imshow('Original Blue', imgFile2)
# Mask the frame to only focus on light panel
masked_data = cv2.bitwise_and(imgFile2, imgFile2, mask=panel)
cv2.imshow('Masked Data', masked_data)

# Covert to HSV to detect color
hsv2 = cv2.cvtColor(masked_data, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV_BLU', hsv2)

# cv2.imwrite('hue.png', hsv2[:,:,0])
# cv2.imwrite('sat.png', hsv2[:,:,1])
# cv2.imwrite('val.png', hsv2[:,:,2])

# loop over the boundaries
for (color, lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(hsv2, lower, upper)
    cv2.imshow('mask', mask)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('Closing', closing)
    #output = cv2.bitwise_and(image, image, mask = mask)
    #closing = cv2.bitwise_not(closing)

    # Detect blobs.
    keypoints = detector.detect(closing)

    #Draw detected blobs as red circles.
    #cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(closing, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Found', im_with_keypoints)

    if keypoints:
        res.append(color)

print ("Colors seen: ", res)


cv2.waitKey(0)
cv2.destroyAllWindows()