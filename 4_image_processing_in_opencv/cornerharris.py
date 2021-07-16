# coding: utf-8
import time

import cv2 as cv
import numpy as np

from mypackage.multiplot import multiplot as mplt


def otsu_threshold(img, min_thre=0, max_thre=255):
    img = cv.GaussianBlur(img, (5, 5), 0)

    img[img < min_thre] = min_thre
    img[img > max_thre] = max_thre

    thre, thre_img = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    print(f'otsu thre:{thre}')
    return thre, thre_img


if __name__ == '__main__':
    imdict = dict()
    img = cv.imread('../mydata/captured_white.png', cv.IMREAD_UNCHANGED)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.fastNlMeansDenoising(gray, None, 5, 7, 21)

    cors = cv.cornerHarris(gray, 2, 3, 0.04)
    print(cors.max())
    img[cors > 0.01 * cors.max()] = [0, 0, 255]
    imdict['img'] = img
    mplt.show(imdict)
