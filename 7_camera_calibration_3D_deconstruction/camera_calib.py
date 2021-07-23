import cv2 as cv
import numpy as np

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)

objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
print(objp[:, :2])
print(objp.shape[::-1])
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
