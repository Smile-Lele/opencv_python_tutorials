# coding: utf-8

import cv2 as cv
import numpy as np

# Load image
cv.samples.addSamplesDataSearchPath('../mydata')
file = cv.samples.findFile('captured_white.png')
if not file:
    raise FileNotFoundError('file not found')

src = cv.imread(file)
src = cv.resize(src, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

img_integral = cv.integral(gray)
h, w = gray.shape[:2]
result = np.zeros((h + 1, w + 1), dtype=np.uint8)
cv.normalize(img_integral, result, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
cv.imshow("result", result)
cv.waitKey()
