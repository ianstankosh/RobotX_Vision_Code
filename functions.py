import numpy as np
import cv2


red = [10, 255, 255, 0, 100, 100]

count = 0

upper_red = [10, 255, 255]
lower_red = [0, 100, 100]

upper_blue = [130, 255, 255]
lower_blue = [110, 50, 50]

upper_green = [70, 255, 255]
lower_green = [50, 100, 100]

upper_black = [180, 255, 40]
lower_black = [0, 0, 0]


def one_color_detection(color, upper, lower):
    # define range of color in HSV(BGR)
    upper_color = np.array(upper)
    lower_color = np.array(lower)
    # Threshold the HSV image to get only input colors
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imwrite("frame%d.jpg" % count, frame)
    cv2.imwrite("mask_" + str(color) + "%d.jpg" % count, mask)
    cv2.imwrite("res_" + str(color) + "%d.jpg" % count, res)


def one_color_detection2(color):
    # define range of color in HSV(BGR)
    upper_color = np.array(color[0:2])
    lower_color = np.array(color[3:5])
    # Threshold the HSV image to get only input colors
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imwrite("frame%d.jpg" % count, frame)
    cv2.imwrite("mask%d.jpg" % count, mask)
    cv2.imwrite("res%d.jpg" % count, res)


def two_color_detection(upper1, lower1, upper2, lower2):
    # define range of color in HSV(BGR)
    upper_color1 = np.array(upper1)
    lower_color1 = np.array(lower1)
    upper_color2 = np.array(upper2)
    lower_color2 = np.array(lower2)
    # Threshold the HSV image to get only input colors
    mask1 = cv2.inRange(hsv, lower_color1, upper_color1)
    mask2 = cv2.inRange(hsv, lower_color2, upper_color2)
    # Bitwise-AND mask and original image
    res1 = cv2.bitwise_and(frame, frame, mask=mask1)
    res2 = cv2.bitwise_and(frame, frame, mask=mask2)

    cv2.imwrite("frame%d.jpg" % count, frame)
    cv2.imwrite("mask%d.jpg" % count, mask1 + mask2)
    cv2.imwrite("res%d.jpg" % count, res1 + res2)


def three_color_detection(upper1, lower1, upper2, lower2, upper3, lower3):
    # define range of color in HSV(BGR)
    upper_color1 = np.array(upper1)
    lower_color1 = np.array(lower1)
    upper_color2 = np.array(upper2)
    lower_color2 = np.array(lower2)
    upper_color3 = np.array(upper3)
    lower_color3 = np.array(lower3)
    # Threshold the HSV image to get only input colors
    mask1 = cv2.inRange(hsv, lower_color1, upper_color1)
    mask2 = cv2.inRange(hsv, lower_color2, upper_color2)
    mask3 = cv2.inRange(hsv, lower_color3, upper_color3)
    # Bitwise-AND mask and original image
    res1 = cv2.bitwise_and(frame, frame, mask=mask1)
    res2 = cv2.bitwise_and(frame, frame, mask=mask2)
    res3 = cv2.bitwise_and(frame, frame, mask=mask3)

    cv2.imwrite("frame%d.jpg" % count, frame)
    cv2.imwrite("mask%d.jpg" % count, mask1 + mask2 + mask3)
    cv2.imwrite("res%d.jpg" % count, res1 + res2 + res3)

def color_properties():
    # Display properties for frames(count) in color
    img = cv2.imread('frame' + str(count) + '.jpg')
    print('frame' + str(count) + ':')
    print('shape:\t' + str(img.shape))
    print('size:\t' + str(img.size))
    print('dtype:\t' + str(img.dtype))
    print('\n')


def gray_properties():
    # Display properties for frames(count) in gray
    print('gray shape:\t' + str(gray.shape))
    print('gray array:\n' + str(gray))
    # Random array value
    print(str(gray[0, 0]) + '\n\n')
    # Save the captured frame as an individual file
    cv2.imwrite("gray%d.jpg" % count, gray)


def hsv_properties():
    # Display properties for frames(count) in hsv
    print('hsv shape:\t' + str(hsv.shape))
    print('hsv array:\n' + str(hsv))
    # Save the captured frame as an individual file
    cv2.imwrite("hsv%d.jpg" % count, hsv)
