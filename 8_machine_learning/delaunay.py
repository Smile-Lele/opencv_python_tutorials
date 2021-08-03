# coding: utf-8

import random

import cv2 as cv
import numpy as np

COLORS = np.random.randint(0, 255, size=(100, 3)).tolist()

# load the path of images
cv.samples.addSamplesDataSearchPath('../mydata')
file = cv.samples.findFile('S000021_P1.png')
if not file:
    raise FileNotFoundError('file not found')

img = cv.imread(file)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

contours, _ = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)
points = [cv.minEnclosingCircle(cnt)[0] for cnt in contours]

points = np.array(points).astype(np.float32)

K = 200
criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 10, 1.0)
ret, labels, centers = cv.kmeans(points, K, None, criteria, attempts=25, flags=cv.KMEANS_RANDOM_CENTERS)

centers = centers.tolist()
for n, c in enumerate(centers):
    pnts = points[labels.ravel() == n]

    rect = cv.boundingRect(pnts)
    subdiv = cv.Subdiv2D(rect)
    [subdiv.insert(p) for p in pnts]

    triangles = subdiv.getTriangleList()
    for t in triangles:
        t = t.reshape(-1, 2).astype(np.int32)
        cv.polylines(img, [t], True, random.choice(COLORS), 1, cv.LINE_8)

    # cv.circle(img, np.uint(c), 8, (0, 0, 255), -1, cv.LINE_8)

cv.imshow('', img)
# cv.imwrite('delaunay.png', img)
key = cv.waitKey() & 0XFF
if key == 27:
    cv.destroyAllWindows()
