import cv2 as cv
import numpy as np


canvas = np.zeros((600, 800), np.uint8)

rect1 = cv.rectangle(canvas.copy(), (50, 50), (200, 500), 255, -1)
rect2 = cv.rectangle(canvas.copy(), (50, 50), (500, 200), 255, -1)

src1, _ = cv.findContours(rect1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
src2, _ = cv.findContours(rect2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

src1 = np.float32(src1[0]).reshape(30, -1)
src2 = np.float32(src2[0]).reshape(30, -1)
print(src1.shape)
print(src2.shape)
dist = cv.batchDistance(src1, src2, cv.CV_32F)
print(dist)

