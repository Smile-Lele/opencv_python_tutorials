# coding:utf-8

import math
import cv2 as cv
from matplotlib import pyplot as plt


def show(imdict):
    """
    This is an adaptive drawing module
    :param imgsdict:
    :return:
    """
    if not isinstance(imdict, dict):
        raise TypeError('param must be type dict()')
    if not imdict:
        print('img dict is empty')

    for index, (title, data) in enumerate(imdict.items()):
        img_num = len(imdict)
        if img_num < 3:
            row = 1
        elif img_num < 7:
            row = 2
        elif img_num < 10:
            row = 3
        else:
            row = 4

        plt.subplot(row, math.ceil(img_num / row), index + 1)

        if len(data.shape) == 1:
            plt.plot(data)
        else:
            img = cv.cvtColor(data, cv.COLOR_BGR2RGB)
            plt.imshow(img, 'gray')
        plt.title(title)
        plt.xticks([]), plt.yticks([])
    plt.show()
