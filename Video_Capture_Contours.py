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
kernel2 = np.ones((40, 40), np.uint8)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Filter by Area.
params.filterByArea = True
params.minArea = 300

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.45
params.maxConvexity = 0.9


def two_color_blob(upper1, lower1, upper2, lower2):
    # define range of yellow in HSV
    upper_hsv1 = np.array(upper1, dtype="uint8")
    lower_hsv1 = np.array(lower1, dtype="uint8")

    # define range of black in HSV
    upper_hsv2 = np.array(upper2, dtype="uint8")
    lower_hsv2 = np.array(lower2, dtype="uint8")

    #cv2.imwrite("frame%d.jpg" % frameCount, frame)
    #img = cv2.imread("frame%d.jpg" % frameCount)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only input colors
    hsv1 = cv2.inRange(hsv, lower_hsv1, upper_hsv1)   # same as mask1 and mask2 in HSV_Processing.py
    hsv2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)

    # Bitwise-OR color1 and color2
    #result = cv2.bitwise_or(hsv1, hsv2)   #only masks image
    res1 = cv2.bitwise_and(frame, frame, mask=hsv1)
    res2 = cv2.bitwise_and(frame, frame, mask=hsv2)

    closing = cv2.morphologyEx(hsv1 + hsv2, cv2.MORPH_CLOSE, kernel1) #hsv1,2 instead of res1,2 b/c can only take B/W
    BuoyBinary = closing  # cv2.bitwise_not(closing)

    #################  Contour Method
    # Find contours (this is a more prmitive way of detecting blobs)
    contours, hierarchy = cv2.findContours(BuoyBinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours2 = None

    distances = []
    angles = []

    # only proceed if at least one contour was found
    if len(contours) > 0:

        # Print the number of contours found
        print ("Number of contours found: ", len(contours))

        # Iterate through all the contours and generate a bounding box
        #   and area of each contour to calculate the Extent.
        #   Also, find the points of the controid and perform height to width exclusion.
        #   This is used to filter which contours to keep.
        for c in contours:
            area = cv2.contourArea(c)

            x, y, w, h = cv2.boundingRect(c)

            rect = cv2.rectangle(BuoyBinary, (x, y), (x + w, y + h), (0, 255, 0), 2)

            extent = area / (w * h)

            M = cv2.moments(c)
            #cx = int(M['m10'] / M['m00'])
            #cy = int(M['m01'] / M['m00'])

            # Height to Width exclusion (adjust for visual distortion in real world)
            hwe = w / h

            # Only keep contours that meet the following area, extent, and height to width exclusion specs.
            # All values based on past projects and real world testing.
            # Bounding box passes the width of the found buoy in pixels
            # if area > 300 and extent >= 0.45 and extent <= 0.9 and hwe <= 1.3:
            #     contours2.append(c)

            #     #The buoy has a fixed diameter (m), the camera has a known field of view
            #     #(which will be converted to radians for calculations) and the size of the
            #     #frame (in pixels) is fixed
            #     d = ((wFrame * diameter) / (w * FoV))*100
            #     a = ((cy - 320) / wFrame * FoV)*180/math.pi
            #     distances.append(d)
            #     angles.append(a)

            print ("Details of contours found:")
            print ("Area: ", area)
            print ("extent: ", extent)
            print ("Moment: ", M)
            #print ("cx: ", cx)
            #print ("cy: ", cy)
            print ("Height to Width exclusion: ", hwe)


            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    else:
        print ("ERROR: No contours found!")

    # return distances, angles
    print ("Distances: ", distances)
    print ("Angles: ", angles)
    #################


    #################  Detector Method
    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector(params)

    # Detect blobs.
    keypoints = detector.detect(BuoyBinary)



    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(BuoyBinary, rect, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #################
    #im_with_contours = cv2.drawContours(BuoyBinary, contours, -1, (0, 255, 0), 3)

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