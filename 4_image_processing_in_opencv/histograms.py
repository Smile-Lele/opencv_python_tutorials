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

src_img = cv.imread('../mydata/calibration.jpg')
src_img = cv.resize(src_img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
img = cv.fastNlMeansDenoising(img, None, 5, 7, 21)
img = cv.GaussianBlur(img, (5, 5), 0)

equ = cv.equalizeHist(img)
res = np.vstack((img, equ))  # stacking images side-by-side
# im_dict['res'] = res

# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_im = clahe.apply(img)
# im_dict['clahe'] = clahe_im

hist = cv.calcHist([clahe_im], [0], mask=None, histSize=[256], ranges=[0, 256])

hist_list = hist.ravel()

# find most adaptive distance and height for find_peaks()
peaks, _ = signal.find_peaks(hist_list, distance=20)
peaks = sorted(peaks, key=lambda x: hist_list[x], reverse=True)
peaks = peaks[:3]
peaks.sort()
print(peaks)
min_distance = min(np.diff(peaks))
min_height = min(hist_list[peaks])

# using min_distance and min_height to find peaks
peaks, properties = signal.find_peaks(hist_list, prominence=1, width=1, distance=min_distance, height=min_height)

plt.plot(hist_list)
plt.plot(peaks, hist_list[peaks], "x")
plt.plot(np.zeros_like(hist_list), "--", color="gray")
plt.vlines(x=peaks, ymin=hist_list[peaks] - properties["prominences"], ymax=hist_list[peaks], color="C1")
plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"], xmax=properties["right_ips"], color="C1")
plt.show()

max_thre = np.nanmean(peaks[1:3])
min_thre = np.nanmean(peaks[:2])

# find rectangle
_, binRecImg = cv.threshold(img, max_thre, 255, cv.THRESH_BINARY)
# binRecImg = cv.medianBlur(binRecImg, 7)
# img_dict['binRecImg'] = binRecImg

# find circle
_, binCirImg = cv.threshold(img, min_thre, 255, cv.THRESH_BINARY_INV)
# binCirImg = cv.medianBlur(binCirImg, 7)
# img_dict['binCirImg'] = binCirImg

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
