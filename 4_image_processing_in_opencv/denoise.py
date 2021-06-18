# coding: utf-8

import cv2 as cv

src = cv.imread('../mydata/share.jpg')
print(type(src))
denoise = cv.fastNlMeansDenoisingColored(src, None, 5, 5, 7, 21)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_im = clahe.apply(denoise)
resize_ = cv.resize(denoise, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
cv.imshow('', resize_)
cv.waitKey()