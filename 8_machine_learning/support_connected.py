# coding: utf-8

import random

import cv2 as cv
import numpy as np
from functools import partial

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


def cart_to_polar(c, c_o=0):
    """
    sort corners by the distance between origin and four points
    :param c:
    :return: magnitude, angle
    """
    magnitude, angle_ = cv.cartToPolar(int(c[0] - c_o[0]), int(c[1] - c_o[1]), angleInDegrees=False)
    return angle_[0], magnitude[0]


points = np.array(points).astype(np.float32)

K = 15
criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 10, 1.0)
ret, labels, centers = cv.kmeans(points, K, None, criteria, attempts=25, flags=cv.KMEANS_RANDOM_CENTERS)

centers = centers.tolist()
for n, c in enumerate(centers):
    pnts = points[labels.ravel() == n]

    pnts_x = sorted(pnts, key=lambda p: (p[0], -p[1]), reverse=True)
    pnts_x = np.uint(pnts_x)
    cv.circle(img, pnts_x[0].tolist(), 8, (0, 0, 255), -1, cv.LINE_8)

    cart_to_polar = partial(cart_to_polar, c_o=pnts_x[0])
    pnts = sorted(pnts, key=cart_to_polar)
    pnts = np.int32(pnts)
    cv.polylines(img, [pnts], True, random.choice(COLORS), 1, cv.LINE_8)

cv.imshow('', img)
# cv.imwrite('delaunay.png', img)
key = cv.waitKey() & 0XFF
if key == 27:
    cv.destroyAllWindows()


