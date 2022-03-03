# coding: utf-8

import cv2 as cv
import numpy as np
from mypackage.imUtils import icv

COLORS = np.random.randint(0, 255, size=(100, 3)).tolist()


# load the path of images
cv.samples.addSamplesDataSearchPath('../mydata')
file = cv.samples.findFile('perspective_white.png')
if not file:
    raise FileNotFoundError('file not found')

img = cv.imread(file)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

minE = cv.cornerMinEigenVal(gray, ksize=9, blockSize=9, borderType=cv.BORDER_DEFAULT)

# TODO: need to research
# minEV = cv.cornerEigenValsAndVecs(gray, ksize=9, blockSize=9, borderType=cv.BORDER_DEFAULT)
# print(minEV)

corners_img = cv.preCornerDetect(gray, ksize=9, borderType=cv.BORDER_DEFAULT)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
dilate_img = cv.morphologyEx(corners_img, cv.MORPH_DILATE, kernel=kernel, iterations=2)

icv.imshow_ex(corners_img)