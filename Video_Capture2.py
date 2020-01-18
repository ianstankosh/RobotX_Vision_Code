import numpy as np
import cv2
import math
import webm


cap = cv2.VideoCapture('/Users/ianstankosh/Desktop/RobotX/field_test_videos/field_test2.5.mp4')
#cap = cv2.VideoCapture(0)
#cap = cv2.imread('/Users/ianstankosh/Destop/442952.jpg')

# Basic way to only focus on sections of interest
# Probably best to make seperate clips of the video when someone has time to do so
frame_per_sec = 30
start_time_in_min = 0
stop_time_in_min = 1
start_frame_count = frame_per_sec * 60 * start_time_in_min
stop_frame_count = frame_per_sec * 60 * stop_time_in_min
frameCount = 0

upper_red = [10, 255, 255]
lower_red = [0, 100, 100]

upper_green = [70, 255, 255]
lower_green = [50, 100, 100]

# Structuring element for morphology
kernel1 = np.ones((12, 12), np.uint8)  # 12by12  square
kernel2 = np.ones((60, 60), np.uint8)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Filter by Area.
params.filterByArea = True
params.minArea = 100

# Filter by Convexity
#params.filterByConvexity = True
#params.minConvexity = 0.45
#params.maxConvexity = 0.9

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 256


def two_color_blob(upper1, lower1, upper2, lower2):
    # define range of yellow in HSV
    upper_hsv1 = np.array(upper1, dtype="uint8")
    lower_hsv1 = np.array(lower1, dtype="uint8")

    # define range of black in HSV
    upper_hsv2 = np.array(upper2, dtype="uint8")
    lower_hsv2 = np.array(lower2, dtype="uint8")

    #cv2.imwrite("frame%d.jpg" % frameCount, frame)
    #img = cv2.imread("frame%d.jpg" % frameCount)

    blur = cv2.GaussianBlur(frame, (3, 3), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only input colors
    mask1 = cv2.inRange(hsv, lower_hsv1, upper_hsv1)   # same as mask1 and mask2 in HSV_Processing.py
    mask2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)

    # Bitwise-OR color1 and color2
    #result = cv2.bitwise_or(hsv1, hsv2)   #only masks image
    res1 = cv2.bitwise_and(frame, frame, mask=mask1)
    res2 = cv2.bitwise_and(frame, frame, mask=mask2)

    # dilate makes the in range areas larger
    mask = cv2.dilate(mask1 + mask2, None, iterations=1)

    closing = cv2.morphologyEx(res1 + res2, cv2.MORPH_CLOSE, kernel2)
    BuoyBinary = closing  # cv2.bitwise_not(closing)

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector(params)

    # Detect blobs.
    #keypoints = detector.detect(BuoyBinary)

    # Detect blobs.
    reversemask = 255 - mask
    keypoints = detector.detect(reversemask)
    if keypoints:
        print "found %d blobs" % len(keypoints)
        if len(keypoints) > 4:
            # if more than four blobs, keep the four largest
            keypoints.sort(key=(lambda s: s.size))
            keypoints = keypoints[0:3]
    else:
        print "no blobs"

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(BuoyBinary, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # cv2.imwrite("frame_%d.jpg" % count, frame)
    #cv2.imshow("binary_%d.jpg" % frameCount, BuoyBinary)
    #cv2.imwrite("keypts_%d.jpg" % count, im_with_keypoints)
    cv2.imshow('keypts_%d.jpg' % frameCount, im_with_keypoints)


while cap.isOpened():
    frameCount += 1

    ret, frame = cap.read()

    if frameCount >= start_frame_count and frameCount < stop_frame_count:
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('frame',gray)
        two_color_blob(upper_red, lower_red, upper_green, lower_green)

    elif frameCount == stop_frame_count:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()