# coding: utf-8

import cv2 as cv
import numpy as np

# src = np.zeros((1080, 1920), np.uint8)
# src[500, :] = 255
# cv.imshow('src', src)
#
# resized_src = cv.resize(src, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
# cv.imshow('resize', resized_src)
# print(cv.countNonZero(resized_src))
# print(resized_src[250, :])
# cv.waitKey()


# k4 = np.zeros((8, 8), np.uint8)
# k4[3:6, 3:6] = 255
# k4[2, 2:7] = 128
# k4[6, 2:7] = 128
# k4[3:6, 2] = 128
# k4[3:6, 6] = 128
# print(k4)
# 
# k4_c = k4[1:, 1:]
# # print(k4_c)
# 
# k4_resize = cv.resize(k4, (6, 6), interpolation=cv.INTER_AREA)
# k4_c_resize = cv.resize(k4_c, (6, 6), interpolation=cv.INTER_AREA)
# 
# # print(k4_resize)
# # print(k4_c_resize)
# 
# k4_resize_c = cv.resize(k4_resize, None, fx=2, fy=2, interpolation=cv.INTER_AREA)
# k4_c_resize_c = cv.resize(k4_c_resize, None, fx=2, fy=2, interpolation=cv.INTER_AREA)
# #
# # print(k4_resize_c)
# # print(k4_c_resize_c)
# 
# add_ = cv.addWeighted(k4_resize_c, 0.5, k4_c_resize_c, 0.5, 0)
# # print(add_)
# 
# add_ = cv.resize(add_, (8, 8), interpolation=cv.INTER_AREA)
# print(add_)


k4 = cv.imread('S000084_P1.png', cv.IMREAD_GRAYSCALE)
print(k4)

k4_crop = k4[1:, 1:]
# print(k4_c)

target_row, target_col = np.around(k4.shape[:2] / np.sqrt(2)).astype(np.uint32)
print(target_col, target_row)
k4_resize = cv.resize(k4, (target_col, target_row), interpolation=cv.INTER_AREA)
k4_crop_resize = cv.resize(k4_crop, (target_col, target_row), interpolation=cv.INTER_AREA)

# print(k4_resize)
# print(k4_c_resize)

k8_resize = cv.resize(k4_resize, None, fx=2, fy=2, interpolation=cv.INTER_AREA)
k4_crop_resize = cv.resize(k4_crop_resize, None, fx=2, fy=2, interpolation=cv.INTER_AREA)
print(k4_crop_resize.shape)
exit()
while True:
    cv.imshow('k8', k4_crop_resize)
    cv.waitKey(2)
    cv.imshow('k8', k8_resize)
    cv.waitKey(2)
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