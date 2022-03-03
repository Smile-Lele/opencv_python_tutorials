from mypackage.imUtils import icv
import cv2 as cv
import numpy as np

img1 = icv.imread_ex('c5.png', cv.IMREAD_GRAYSCALE)
img2 = icv.imread_ex('c7.png', cv.IMREAD_GRAYSCALE)

res1 = cv.meanStdDev(img1)
res2 = cv.meanStdDev(img2)
print(res1)
print(res2)