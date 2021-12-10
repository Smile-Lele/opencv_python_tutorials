import cv2 as cv
import numpy as np

src = np.array([[1, 2, 3], [2, 3, 4], [4, 5, 6]])
mask = np.array([[255, 255, 255], [255, 255, 255], [255, 255, 255]])

res = cv.bitwise_and(src, mask)
print(res)