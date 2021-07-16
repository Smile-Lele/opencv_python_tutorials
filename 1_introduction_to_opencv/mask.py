# coding: utf-8
import time

import cv2 as cv
import numpy as np
from mypackage.multiplot import multiplot as mplt
from numpy import ma

if __name__ == '__main__':
    array = np.array([[1, 2, 3], [4, 5, 6]])
    print(array)
    print(f'array:{array.shape},{array.ndim},{array.dtype}')

    t1 = time.time()
    for i in range(10000):
        ma_a = ma.masked_greater(array, 4)
        ma_a.filled(4)
    print(time.time() -t1)

    t2 = time.time()
    for i in range(10000):
        array[array>4] = 4
    print(time.time() - t2)