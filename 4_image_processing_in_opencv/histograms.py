# coding: utf-8
import random

from scipy import signal

import cv2 as cv
import numpy as np
from mypackage.multiplot import multiplot as mplt
from matplotlib import pyplot as plt

im_dict = dict()
COLORS = [(48, 48, 255),
          (0, 165, 255),
          (0, 255, 0),
          (255, 255, 0),
          (147, 20, 255),
          (144, 238, 144)]

src_img = cv.imread('../mydata/calib_2.bmp')
src_img = cv.resize(src_img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
img = cv.fastNlMeansDenoising(img, None, 5, 7, 21)
img = cv.GaussianBlur(img, (5, 5), 0)

# equalize hist to increase the contrast of images
equ = cv.equalizeHist(img)
# stacking images side-by-side
res = np.vstack((img, equ))

# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_im = clahe.apply(img)

hist = cv.calcHist([clahe_im], [0], mask=None, histSize=[256], ranges=[0, 256])

hist = hist.squeeze()
hist_list = hist.tolist()
MAX_GRAYSCALE = hist_list.index(max(hist_list))
print(MAX_GRAYSCALE)

# to draw nice histogram, just for fun, not important
peaks, properties = signal.find_peaks(hist, prominence=1, width=1, distance=50, height=100)
plt.plot(hist)
plt.plot(peaks, hist[peaks], "x")
plt.plot(np.zeros_like(hist), "--", color="gray")
plt.vlines(x=peaks, ymin=hist[peaks] - properties["prominences"], ymax=hist[peaks], color="C1")
plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"], xmax=properties["right_ips"], color="C1")
plt.show()

# find rectangle
binRecImg = img.copy()
binRecImg[binRecImg <= MAX_GRAYSCALE - 30] = MAX_GRAYSCALE
_, binRecImg = cv.threshold(binRecImg, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
# im_dict['binRecImg'] = binRecImg

# find circle
binCirImg = img.copy()
binCirImg[binCirImg >= MAX_GRAYSCALE + 30] = MAX_GRAYSCALE
_, binCirImg = cv.threshold(binCirImg, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
# im_dict['binCirImg'] = binCirImg

# concat img
concat_img = cv.bitwise_or(binCirImg, binRecImg)
im_dict['concat_img'] = concat_img

# find contours
contours, _ = cv.findContours(concat_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for idx, _ in enumerate(contours):
    cv.drawContours(src_img, contours, idx, random.choice(COLORS), 2)
im_dict['src'] = src_img

mplt.show(im_dict)

# key = cv.waitKey() & 0XFF
# cv.destroyAllWindows()
