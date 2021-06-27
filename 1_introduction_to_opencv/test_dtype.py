# coding: utf-8

import cv2 as cv
import numpy as np

src = cv.imread('../mydata/messi5.jpg')
src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)



print('channel:', src.shape)
print('ndim:', src.ndim)
print('dtype:', src.dtype)


