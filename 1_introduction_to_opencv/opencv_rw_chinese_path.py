import os

import cv2 as cv
import numpy as np

path = 'D:/opencv中文路径'
# read image from chinese path
img = cv.imdecode(np.fromfile(os.path.join(path, 'test.png'), dtype=np.uint8), -1)

# write image to chinese path
cv.imencode('.png', img)[1].tofile(os.path.join(path, 'test_w.png'))
