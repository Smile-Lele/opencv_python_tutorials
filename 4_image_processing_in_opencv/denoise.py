# coding: utf-8

import cv2 as cv
from mypackage.multiplot import multiplot as mplt
import numpy as np

imdict = dict()
filepath = '../mydata/messi5.jpg'
# imdecode can deal with the error lead by chinese path
src = cv.imdecode(np.fromfile(filepath, dtype=np.uint8), cv.IMREAD_UNCHANGED)
# imdict['src'] = src
src = cv.resize(src, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
denoise = cv.fastNlMeansDenoisingColored(src, None, 9, 9, 7, 21)
imdict['denoise'] = denoise
cv.normalize(denoise, denoise, 0, 255, cv.NORM_MINMAX)
imdict['norm'] = denoise
resize_im = cv.resize(denoise, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
cv.imwrite('../mydata/messi5_E.jpg', resize_im)
mplt.show(imdict)
