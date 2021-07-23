# coding: utf-8

import cv2 as cv
import numpy as np

from mypackage.multiplot import multiplot as mplt

prev = np.array([[10, 5, 10], [8, 3, 6], [12, 8, 12]], dtype=np.float32)
LINEAR_jet = cv.resize(prev, (9, 9), interpolation=cv.INTER_LINEAR)
LINEAR_EXACT_jet = cv.resize(prev, (9, 9), interpolation=cv.INTER_LINEAR_EXACT)
AREA_jet = cv.resize(prev, (9, 9), interpolation=cv.INTER_AREA)
CUBIC_jet = cv.resize(prev, (9, 9), interpolation=cv.INTER_CUBIC)
LANCZOS_jet = cv.resize(prev, (9, 9), interpolation=cv.INTER_LANCZOS4)
BITS_jet = cv.resize(prev, (9, 9), interpolation=cv.INTER_BITS)
print(CUBIC_jet.squeeze().T)

imdict = dict()
imdict['LINEAR_jet'] = LINEAR_jet
imdict['LINEAR_EXACT_jet'] = LINEAR_EXACT_jet
imdict['AREA_jet'] = AREA_jet
imdict['CUBIC_jet'] = CUBIC_jet
imdict['LANCZOS_jet'] = LANCZOS_jet
imdict['BITS_jet'] = BITS_jet
mplt.show(imdict)
