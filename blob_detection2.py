# This program runs HSV_Processing.py and uses its frames in order to close masked images into blobs
#
import numpy as np
import cv2
import math
import HSV_Processing

# Structuring element for morphology
kernel = np.ones((12, 12), np.uint8)  # 12by12  square

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Filter by Area.
params.filterByArea = True
params.minArea = 300

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.45
params.maxConvexity = 0.9

# Used to keep track of the frames
count = 0


def two_color_blob(upper1, lower1, upper2, lower2):
    # define range of yellow in HSV
    upper_hsv1 = np.array(upper1, dtype="uint8")
    lower_hsv1 = np.array(lower1, dtype="uint8")

    # define range of black in HSV
    upper_hsv2 = np.array(upper2, dtype="uint8")
    lower_hsv2 = np.array(lower2, dtype="uint8")

    img = cv2.imread('frame%d.jpg' % count)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only input colors
    hsv1 = cv2.inRange(hsv, lower_hsv1, upper_hsv1)
    hsv2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)

    # Bitwise-OR yellow and black
    result = cv2.bitwise_or(hsv1, hsv2)

    closing = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    BuoyBinary = closing  # cv2.bitwise_not(closing)

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector(params)

    # Detect blobs.
    keypoints = detector.detect(BuoyBinary)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(BuoyBinary, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # cv2.imwrite("frame_%d.jpg" % count, frame)
    cv2.imwrite("binary_%d.jpg" % count, BuoyBinary)
    cv2.imwrite("keypts_%d.jpg" % count, im_with_keypoints)
    cv2.imshow('keypts_%d.jpg' % count, im_with_keypoints)


def one_buoy(upper, lower):
    # define range of yellow in HSV
    upper_hsv = np.array(upper, dtype="uint8")
    lower_hsv = np.array(lower, dtype="uint8")

    img = cv2.imread('frame3.jpg')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only input colors
    ylw = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Perform closing on image
    closing = cv2.morphologyEx(ylw, cv2.MORPH_CLOSE, kernel)
    BuoyBinary = closing  # cv2.bitwise_not(closing)

    # Find contours (this is a more prmitive way of detecting blobs)
    contours, hierarchy = cv2.findContours(BuoyBinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2 = None

    diameter = 0.203
    FoV = 62 * (math.pi) / 180
    wFrame = 640
    distances = None
    angles = None

    # ww
    print(diameter)
    print(FoV)
    print(wFrame)
    print(distances)
    print(angles)

    # only proceed if at least one contour was found
    if len(contours) > 0:

        # Print the number of contours found
        print len(contours)

        # Iterate through all the contours and generate a bounding box
        #   and area of each contour to calculate the Extent.
        #   Also, find the points of the controid and perform height to width exclusion.
        #   This is used to filter which contours to keep.
        for c in contours:

            area = cv2.contourArea(c)
            cnt = contours[0]
            x, y, w, h = cv2.boundingRect(cnt)
            extent = area / (w * h)

            M = cv2.moments(c)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Height to Width exclusion (adjust for visual distortion in real world)
            hwe = w / h

            # Only keep contours that meet the following area, extent, and height to width exclusion specs.
            # All values based on past projects and real world testing.
            # Bounding box passes the width of the found buoy in pixels
            if area > 300 and extent >= 0.45 and extent <= 0.9 and hwe <= 1.3:
                contours2.append(c)

                # The buoy has a fixed diameter (m), the camera has a known field of view
                # (which will be converted to radians for calculations) and the size of the
                # frame (in pixels) is fixed
                distances.append(((wFrame * diameter) / (w * FoV)) * 100)
                angles.append(((cy - 320) / wFrame * FoV) * 180 / math.pi)

                # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            print(area)
            print(M)
            print(cx)
            print(cy)

    return distances, angles


while count < 4:
    two_color_blob(HSV_Processing.upper_yellow, HSV_Processing.lower_yellow, HSV_Processing.upper_black,
                   HSV_Processing.lower_black)
    count += 1

one_buoy(HSV_Processing.upper_yellow, HSV_Processing.lower_yellow)

cv2.destroyAllWindows()

