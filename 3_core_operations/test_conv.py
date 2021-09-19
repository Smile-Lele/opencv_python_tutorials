import os
import random

import cv2 as cv
import numpy as np
from mypackage.imUtils import impreprocessing as impre

COLORS = np.random.randint(150, 255, size=(100, 3)).tolist()

# load the path of images
cv.samples.addSamplesDataSearchPath('../mydata')
file = cv.samples.findFile('captured_white_1.png')
if not file:
    raise FileNotFoundError('file not found')

img = cv.imread(file)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

kernel_dx = np.array([[-3, 0, 3],
                      [-3, 0, 3],
                      [-3, 0, 3]])
dx = cv.filter2D(gray, cv.CV_32FC1, kernel_dx)
dx = cv.convertScaleAbs(dx)
_, dx = impre.otsu_threshold(dx, visibility=False)

kernel_dy = np.array([[-3, -3, -3],
                      [0, 0, 0],
                      [3, 3, 3]])
dy = cv.filter2D(gray, cv.CV_32FC1, kernel_dy)
dy = cv.convertScaleAbs(dy)
_, dy = impre.otsu_threshold(dy, visibility=False)

concat = cv.addWeighted(dx, 1, dy, 1, 0)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
concat = cv.morphologyEx(concat, cv.MORPH_DILATE, kernel=kernel, iterations=1)

# find external contour
contours, _ = cv.findContours(concat, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)

# sort all of contours detected to
contours = sorted(contours, key=lambda ct: cv.arcLength(ct, True), reverse=True)

cnt = contours[0]
epsilon = 0.1 * cv.arcLength(cnt, True)
approx = cv.approxPolyDP(cnt, epsilon, True)
# print(approx.squeeze().tolist())
assert len(approx) == 4, 'approx res should be 4'

corners = approx.squeeze()
for corner in corners:
    corner = tuple(corner)
    # cv.circle(src_img, corner, 6, random.choice(COLORS), -1)
    cv.drawMarker(img, corner, random.choice(COLORS), markerType=cv.MARKER_CROSS, markerSize=20, thickness=2)

cv.imshow('win', img)
cv.waitKey()
cv.destroyWindow('win')
