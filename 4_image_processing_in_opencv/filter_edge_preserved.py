# coding: utf-8

import cv2 as cv
import numpy as np
from mypackage.multiplot import multiplot as mplt

# load the path of images
cv.samples.addSamplesDataSearchPath('../mydata')
file = cv.samples.findFile('captured_white.png')
if not file:
    raise FileNotFoundError('file not found')

img = cv.imread(file)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

imdict = dict()
# Gaussian
gaussian = cv.GaussianBlur(gray, (3, 3), 0)
imdict['gaussian'] = gaussian

# bilateralFilter
bilateral = cv.bilateralFilter(gray, 9, 85, 85)
imdict['bilateral'] = bilateral

# pyrMeanShiftFiltering
meanshift = cv.pyrMeanShiftFiltering(img, 15, 30, termcrit=(cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 5, 1))
imdict['meanshift'] = meanshift

# edgePreservingFilter
edgepreserve = cv.edgePreservingFilter(img, sigma_s=50, sigma_r=0.4, flags=cv.RECURS_FILTER)
imdict['edgepreserve'] = edgepreserve

mplt.show(imdict)
