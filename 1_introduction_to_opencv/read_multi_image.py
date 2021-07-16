import glob
import os

import cv2 as cv
import numpy as np

imdict = dict()
file_path = 'D:/MyData/1'

files = glob.glob(os.path.join(file_path, '*.png'))
if not files:
    raise FileNotFoundError('file not found')

imgs = list(map(lambda file: cv.imread(file, cv.IMREAD_GRAYSCALE), files))
sum_ = np.zeros(imgs[0].shape, np.uint8)
for img in imgs:
    sum_ = cv.add(sum_, img)
    cv.imshow('', sum_)
    key = cv.waitKey(5) & 0XFF
    if key == 27:
        cv.destroyAllWindows()
        break
