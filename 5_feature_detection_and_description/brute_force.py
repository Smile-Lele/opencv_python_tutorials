# coding: utf-8

import cv2 as cv
from mypackage.multiplot import multiplot as mplt
import numpy as np

if __name__ == '__main__':

    imdict = dict()
    img = cv.imread('../mydata/box.png', cv.IMREAD_GRAYSCALE)
    background = cv.imread('../mydata/box_in_scene.png', cv.IMREAD_GRAYSCALE)

    sift = cv.SIFT_create()
    kp1, res1 = sift.detectAndCompute(img, None)
    kp2, res2 = sift.detectAndCompute(background, None)

    bf = cv.BFMatcher(crossCheck=True)
    matches = bf.match(res1, res2)
    matches = sorted(matches, key=lambda x:x.distance)

    des = cv.drawMatches(img, kp1, background, kp2, matches[:10], None, flags=2)
    imdict['des'] = des

    bf2 = cv.FlannBasedMatcher()
    matches = bf2.knnMatch(res1, res2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    
    knn = cv.drawMatchesKnn(img, kp1, background, kp2, good, None, flags=2)
    imdict['knn'] = knn

    mplt.show(imdict)