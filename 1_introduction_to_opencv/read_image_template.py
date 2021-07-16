# coding: utf-8

import cv2 as cv

# load the path of images
cv.samples.addSamplesDataSearchPath('../mydata')
file = cv.samples.findFile('drop.png')
if not file:
    raise FileNotFoundError('file not found')

img = cv.imread(file)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('', gray)
key = cv.waitKey(1000) & 0XFF
if key == 27:
    cv.destroyAllWindows()