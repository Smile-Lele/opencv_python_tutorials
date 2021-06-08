# coding: utf-8

import numpy as np
import cv2 as cv
from mypackage.multiplot import multiplot as mplt


def main():
    img = cv.imread('../mydata/drop.png')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    imdict = dict()
    imdict['thresh'] = thresh

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    imdict['opening'] = opening

    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    imdict['sure_bg'] = sure_bg

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    imdict['dist_transform'] = dist_transform
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    imdict['sure_fg'] = sure_fg

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    imdict['unknown'] = unknown

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv.watershed(img, markers)
    imdict['jet_markers'] = markers

    img[markers == -1] = [0, 0, 255]
    imdict['img'] = img
    mplt.show(imdict)


if __name__ == '__main__':
    main()
