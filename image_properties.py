import numpy as np
import cv2
from HSV_Processing import gray, hsv, frame, shape, size, dtype, count

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


color_properties()
gray_properties()
hsv_properties()