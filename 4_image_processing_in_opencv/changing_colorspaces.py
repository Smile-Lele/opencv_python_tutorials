# coding: utf-8
import os
import numpy as np
import cv2 as cv
from mypackage.multiplot import multiplot as mplt


def main():
    # using dir() to look at inner functions
    # flags = [i for i in dir(cv) if i.startswith('COLOR_')]
    # print(flags)

    # how to track HSV value
    green = np.uint8([[[0, 0, 255]]])
    hsv_green = cv.cvtColor(green, cv.COLOR_BGR2HSV)
    hsv_g = hsv_green.ravel()
    print(hsv_g)

    filename = os.path.join('../mydata', 'animal.jpg')
    src_img = cv.imread(filename)
    hsv_img = cv.cvtColor(src_img, cv.COLOR_BGR2HSV)

    # define range of green color in HSV
    lower_green = np.array([hsv_g[0] - 20, 50, 50])
    higher_green = np.array([hsv_g[0] + 20, 255, 255])

    # threshold the HSV image to get only green colors
    mask = cv.inRange(hsv_img, lower_green, higher_green)

    # Bitwise-AND mask and original image
    res = cv.bitwise_and(src_img, src_img, mask=mask)

    imdict = dict()
    imdict['scr'] = src_img
    imdict['mask'] = mask
    imdict['res'] = res

    mplt.show(imdict)
    # key = cv.waitKey() & 0xFF
    # cv.destroyAllWindows()


if __name__ == '__main__':
    main()
