import cv2 as cv
import numpy as np
from mypackage import icv

imdict = {}
canvas = np.zeros((600, 800), np.uint8)

rect1 = cv.rectangle(canvas.copy(), (50, 50), (200, 500), 255, -1)
rect2 = cv.rectangle(canvas.copy(), (50, 50), (500, 200), 255, -1)

rect1_cnt, _ = cv.findContours(rect1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)
rect2_cnt, _ = cv.findContours(rect2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)

# TODO: decide whether gray2color is needed for drawContours
rect1 = icv.cvtGray2BGR(rect1)
rect2 = icv.cvtGray2BGR(rect2)

# find minRectArea
minRect1 = cv.minAreaRect(rect1_cnt[0])
minRect2 = cv.minAreaRect(rect2_cnt[0])

# get intersection between two rotated rectangles
interType, points = cv.rotatedRectangleIntersection(minRect1, minRect2)
points = points.reshape(4, 2).astype(np.int32)

cv.drawContours(rect1, rect1_cnt, -1, (0, 255, 0), 2)
cv.drawContours(rect2, rect2_cnt, -1, (0, 0, 255), 2)
imdict['rect1'] = rect1
imdict['rect2'] = rect2

add_ = cv.scaleAdd(rect1, 1, rect2, 1)
imdict['canvas'] = add_

# draw polygon
inter = cv.fillPoly(add_, [points], (0, 255, 255))
imdict['inter'] = inter


icv.implot_ex(imdict)
