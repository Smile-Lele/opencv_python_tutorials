# coding: utf-8

import cv2 as cv
import numpy as np
from mypackage.multiplot import multiplot as mplt

COLORS = np.random.randint(0, 255, size=(100, 3)).tolist()


# load the path of images
cv.samples.addSamplesDataSearchPath('../mydata')
file = cv.samples.findFile('S000021_P1.png')
if not file:
    raise FileNotFoundError('file not found')

imdict = dict()
img = cv.imread(file)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
re_gray = cv.resize(gray, None, fx=0.01, fy=0.01, interpolation=cv.INTER_AREA)
print(re_gray)
imdict['re_gray'] = re_gray

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
print(kernel)
imdict['kernel'] = kernel


morph_gray = cv.morphologyEx(re_gray, cv.MORPH_DILATE, kernel)
print(morph_gray)
imdict['morph_gray'] = morph_gray


diff = morph_gray - re_gray
print(diff)
imdict['diff'] = diff

mplt.show(imdict)