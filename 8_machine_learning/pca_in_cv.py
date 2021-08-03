# coding: utf-8

import cv2 as cv
import numpy as np
from math import atan2, cos, sin, sqrt, pi
import random


COLORS = np.random.randint(0, 255, size=(100, 3)).tolist()



def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)


def getOrientation(pts, img):
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    # Perform PCA analysis
    mean = np.empty(0)
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean, 0.99)

    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))

    # Draw the principal components
    cv.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (
        cntr[0] + 0.002 * eigenvectors[0, 0] * eigenvalues[0, 0],
        cntr[1] + 0.002 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (
        cntr[0] - 0.002 * eigenvectors[1, 0] * eigenvalues[1, 0],
        cntr[1] - 0.002 * eigenvectors[1, 1] * eigenvalues[1, 0])
    drawAxis(img, cntr, p1, random.choice(COLORS), 1)
    drawAxis(img, cntr, p2, random.choice(COLORS), 1)

    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians

    return angle


def main():
    # Load image
    cv.samples.addSamplesDataSearchPath('../mydata')
    file = cv.samples.findFile('captured_white.png')
    if not file:
        raise FileNotFoundError('file not found')

    src = cv.imread(file)
    src = cv.resize(src, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # Convert image to binary
    _, bw = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # Find all the contours in the thresholded image
    contours, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv.contourArea(c)
        # Ignore contours that are too small or too large
        if area < 1e2:
            continue

        # Draw each contour only for visualisation purposes
        cv.drawContours(src, contours, i, (0, 0, 255), 2)
        # Find the orientation of each shape
        angle = getOrientation(c, src)
        print(angle)

    cv.imshow('output', src)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
