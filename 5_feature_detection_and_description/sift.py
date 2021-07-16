# coding: utf-8

import cv2 as cv
from mypackage.multiplot import multiplot as mplt
import numpy as np

if __name__ == '__main__':

    imdict = dict()
    img = cv.imread('../mydata/home.jpg', cv.IMREAD_UNCHANGED)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imdict['gray'] = gray
    sift = cv.SIFT_create()
    kp, res = sift.detectAndCompute(gray, None)
    print(res.shape)
    img = cv.drawKeypoints(gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    imdict['img'] = img
    mplt.show(imdict)