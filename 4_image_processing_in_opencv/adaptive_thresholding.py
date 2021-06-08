# coding:utf-8

import cv2 as cv
from matplotlib import pyplot as plt
import os

filename = os.path.join('../mydata', 'calibration.jpg')
img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
img = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
img = cv.fastNlMeansDenoising(img, None, h=5, templateWindowSize=7, searchWindowSize=21)
# img = cv.medianBlur(img, 5)
# img = cv.GaussianBlur(img, (13,13), 0)
img = cv.bilateralFilter(img, 9, 75, 75)
ret, th1 = cv.threshold(img, 35, 255, cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, 2)
th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
