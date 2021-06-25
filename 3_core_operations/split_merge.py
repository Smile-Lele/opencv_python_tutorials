# coding: utf-8

import cv2 as cv
from mypackage.multiplot import multiplot as mplt

src = cv.imread('../mydata/messi5.jpg')
print(type(src))
denoise = cv.fastNlMeansDenoisingColored(src, None, 5, 5, 7, 21)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
mv = cv.split(denoise)
mv[0] = clahe.apply(mv[0])
mv[1] = clahe.apply(mv[1])
mv[2] = clahe.apply(mv[2])
cla_im = cv.merge(mv)
imdict = dict()
imdict['cla_im'] = cla_im
mplt.show(imdict)