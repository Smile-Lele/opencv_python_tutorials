# coding: utf-8

import random

import cv2 as cv
import numpy as np

COLORS = np.random.randint(50, 255, size=(100, 3)).tolist()

# load the path of images
cv.samples.addSamplesDataSearchPath('../mydata')
file = cv.samples.findFile('S000022_P2.png')
if not file:
    raise FileNotFoundError('file not found')

img = cv.imread(file)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

contours, _ = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)
points = [cv.minEnclosingCircle(cnt)[0] for cnt in contours]

points = np.array(points).astype(np.float32)

K = len(points) // 6
criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 10, 1.0)
ret, labels, centers = cv.kmeans(points, K, None, criteria, attempts=25, flags=cv.KMEANS_RANDOM_CENTERS)
centers = centers.tolist()

# draw centers
corners = [cv.KeyPoint(c[0], c[1], 1) for c in centers]
img = cv.drawKeypoints(img, corners, None, color=random.choice(COLORS))

for n in labels:
    pnts = points[labels.ravel() == n]

    rect = cv.boundingRect(pnts)
    subdiv = cv.Subdiv2D(rect)
    subdiv.insert(pnts.tolist())

    triangles = subdiv.getTriangleList()
    for t in triangles:
        t = t.reshape(-1, 2).astype(np.int32)
        cv.polylines(img, [t], True, random.choice(COLORS), 2, cv.LINE_8)


img = cv.resize(img, None, fx=0.8, fy=0.8, interpolation=cv.INTER_AREA)
cv.imshow('', img)
# cv.imwrite('delaunay.png', img)
key = cv.waitKey() & 0XFF
if key == 27:
    cv.destroyAllWindows()
