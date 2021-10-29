# coding: utf-8

import cv2 as cv
import numpy as np

k4 = cv.imread('S000109_P1.png', cv.IMREAD_GRAYSCALE)

target_row, target_col = (1080, 1920)

k1_tl = cv.resize(k4, (target_col, target_row), interpolation=cv.INTER_AREA)
k1_tr = cv.resize(k4[:, :-1], (target_col, target_row), interpolation=cv.INTER_AREA)
k1_bl = cv.resize(k4[:-1, :], (target_col, target_row), interpolation=cv.INTER_AREA)
k1_br = cv.resize(k4[:-1, :-1], (target_col, target_row), interpolation=cv.INTER_AREA)
cv.imwrite('k1_tl.png', k1_tl)
cv.imwrite('k1_tr.png', k1_tr)
cv.imwrite('k1_bl.png', k1_bl)
cv.imwrite('k1_br.png', k1_br)




k1_tl = cv.resize(k1_tl, None, fx=2, fy=2, interpolation=cv.INTER_AREA)
k1_tr = cv.resize(k1_tr, None, fx=2, fy=2, interpolation=cv.INTER_AREA)
k1_bl = cv.resize(k1_bl, None, fx=2, fy=2, interpolation=cv.INTER_AREA)
k1_br = cv.resize(k1_br, None, fx=2, fy=2, interpolation=cv.INTER_AREA)



addweight_img = k1_tl / 4 + k1_tr / 4 + k1_bl / 4 + k1_br / 4
addweight_img = np.uint8(addweight_img)

cv.imwrite('4k_1k.png', addweight_img)

# #
# # print(k4_resize_c)
# # print(k4_c_resize_c)
#
# add_ = cv.addWeighted(k4_resize_c, 0.5, k4_c_resize_c, 0.5, 0)
# # print(add_)
#
# add_ = cv.resize(add_, (k4.shape[1], k4.shape[0]), interpolation=cv.INTER_AREA)
# cv.imshow('', add_)
# cv.imwrite('P.png', add_)
# cv.waitKey()
