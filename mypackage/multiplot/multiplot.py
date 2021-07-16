# coding:utf-8

import math
import cv2 as cv
from matplotlib import pyplot as plt


def show(imdict):
    """
    This is an adaptive drawing module
    :param imdict:
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

        if data.ndim == 1:
            plt.plot(data)
        else:
            if data.ndim == 3:
                data = cv.cvtColor(data, cv.COLOR_BGR2RGB)

            cmap = ['gray', 'jet']['jet' in title]
            plt.imshow(data, cmap)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
    plt.show()
