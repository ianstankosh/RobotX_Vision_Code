import numpy as np
import cv2
import math

# Structuring element for morphology
kernel1 = np.ones((12, 12), np.uint8)  # 12by12  square
kernel2 = np.ones((20, 20), np.uint8)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Filter by Area.
params.filterByArea = True
params.minArea = 300

# Change thresholds
#params.minThreshold = 0
#params.maxThreshold = 256

# Filter by Convexity
#params.filterByConvexity = True
#params.minConvexity = 0.45
#params.maxConvexity = 0.9

#params.filterByColor = True
#params.blobColor = 0

# Used to keep track of the frames
count = 0


def black_and_yellow_buoys(frame):
    # define range of yellow in HSV
    yellow_upper_hsv = np.array([55, 255, 255], dtype="uint8")
    yellow_lower_hsv = np.array([30, 130, 150], dtype="uint8")

    # define range of black in HSV
    black_upper_hsv = np.array([10, 255, 255], dtype="uint8")
    black_lower_hsv = np.array([0, 100, 100], dtype="uint8")

    # Threshold the HSV image to get only input colors
    ylw = cv2.inRange(hsv, yellow_lower_hsv, yellow_upper_hsv)
    blk = cv2.inRange(hsv, black_lower_hsv, black_upper_hsv)

    # Bitwise-OR yellow and black
    result = cv2.bitwise_or(ylw, blk)

    closing = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel2)
    BuoyBinary = closing  # cv2.bitwise_not(closing)

    ####
    white = np.array([0, 0, 255], dtype="uint8")
    whiteClosing = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel2)
    whiteFinal = whiteClosing
    ####

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector(params)

    # Detect blobs.
    keypoints = detector.detect(frame)

    # Draw detected blobs as red circles (0, 0, 255)BGR
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(BuoyBinary, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imwrite("frame%d.jpg" % count, frame)
    cv2.imwrite("binary_%d.jpg" % count, BuoyBinary)
    cv2.imshow("binary_%d.jpg" % count, BuoyBinary)
    cv2.imwrite("keypts_%d.jpg" % count, im_with_keypoints)
    cv2.imshow('keypts', im_with_keypoints)


def red_buoys(frame):
    # define range of yellow in HSV
    red_upper_hsv = np.array([10, 255, 255], dtype="uint8")
    red_lower_hsv = np.array([0, 100, 100], dtype="uint8")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV', hsv)

    # Threshold the HSV image to get only input colors
    red = cv2.inRange(hsv, red_lower_hsv, red_upper_hsv)
    cv2.imshow('Threshold', red)

    # Perform closing on image
    closing = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel1)
    BuoyBinary = closing  # cv2.bitwise_not(closing)

    # Find contours (this is a more prmitive way of detecting blobs)
    contours, hierarchy = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Values derived from previous work
    diameter = 0.203
    FoV = 62 * (math.pi) / 180
    wFrame = 640
    distances = []
    angles = []

    # only proceed if at least one contour was found
    if len(contours) > 0:

        # Print the number of contours found
        print ("Number of contours found: %d" % len(contours))

        # Iterate through all the contours and generate a bounding box
        #   and area of each contour to calculate the Extent.
        #   Also, find the points of the controid and perform height to width exclusion.
        #   This is used to filter which contours to keep.
        for c in contours:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            extent = area / (w * h)

            cv2.drawContours(red, contours, -1, (0, 0, 255), 3)
            #cv2.rectangle(contours, (x, y), (x + w, y + h), (0, 255, 0), 2)


            M = cv2.moments(c)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

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
            print ("Area: %d" % area)
            print ("extent: %d" % extent)
            print ("Moment: %s" % M)
            print ("cx: %d" % cx)
            print ("cy: %d" % cy)
            print ("Height to Width exclusion: %d" % hwe)

    else:
        print ("ERROR: No contours found!")

    cv2.imshow('BuoyBinary', red)

    # return distances, angles
    print ("Distances: %s" % distances)
    print ("Angles: %s" % angles)


imgFile = cv2.imread('/Users/ianstankosh/Desktop/red.png')
cv2.imshow('Original', imgFile)

red_buoys(imgFile)

#black_and_yellow_buoys(imgFile)
#black_and_yellow_buoys(imgFile)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # VideoCapture using the only available camera
# cap = cv2.VideoCapture(0)

# # While camera opened...
# while cap.isOpened() and count < 4:

#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # 'Live' operation on frame
#     #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Convert BGR to HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # Image closing
#     #closing = cv2.morphologyEx(hsv, cv2.MORPH_CLOSE, kernel)

#     black_and_yellow_buoys()

#     # Save the captured frame as an individual file
#     # --> can be used for processing frame-by-frame..,
#     #cv2.imwrite("frame%d.jpg" % count, frame)
#     count = count + 1   # Increment count

#     # 'Live' display of processed frames
#     cv2.imshow('frame',hsv)
#     #cv2.imshow('frame',hsv)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

# Display properties for frame #5
# img = cv2.imread('frame5.jpg')
# print (img.shape)
# print (img.size)
# print (img.dtype)