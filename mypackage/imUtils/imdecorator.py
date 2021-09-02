# coding: utf-8

import cv2 as cv
import numpy as np


def im_decorate(src, pendant, position: np.float32, scale=1, angle=0, transparency=1):
    """
        This method is to decorate source image using given image.
    :param src: image needed to decorate
    :param pendant: single element used to decorate src
    :param position: the place where the center of pendant should be placed
    :param scale: scale pendant
    :param angle: rotate angle, positive is clockwise-counter
    :param transparency: adjust transparency
    :return: a new image that is decorated by pendant
    """
    assert src.ndim == pendant.ndim, f'src.dim{src.ndim} != pendant.dim{pendant.ndim}'

    # get center of src
    row, col = src.shape[:2]
    center = np.divide((col, row), 2).astype(np.float32)  # center(x, y)

    print(f'Position:{position}', end='')

    # scale
    if 0 < scale != 1:
        print(f' | Scale:{scale}', end='')
        interpolation = cv.INTER_CUBIC if scale > 1 else cv.INTER_AREA
        pendant = cv.resize(pendant, None, fx=scale, fy=scale, interpolation=interpolation)

    # get center of pendant
    row_p, col_p = pendant.shape[:2]
    center_p = np.divide((col_p, row_p), 2).astype(np.float32)  # center(x, y)

    # rotate, first step is to move pendant to the center of src, in order to avoid
    # lost pixels on the edges while rotating
    if angle != 0:
        print(f' | Angle:{angle}', end='')
        tx, ty = center - center_p
        tran_mat = np.matrix([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        pendant = cv.warpAffine(pendant, tran_mat, (col, row), cv.BORDER_TRANSPARENT)

        rot_mat = cv.getRotationMatrix2D(center, angle, scale=1)
        pendant = cv.warpAffine(pendant, rot_mat, (col, row), cv.BORDER_TRANSPARENT)

    # translate
    tx, ty = position - center
    tran_mat = np.matrix([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    pendant = cv.warpAffine(pendant, tran_mat, (col, row), cv.BORDER_TRANSPARENT)

    # concat
    print(f' | Transparency:{transparency}')
    target = cv.addWeighted(src, 1, pendant, transparency, 0)
    return target


# src = cv.imread('captured_white.png')
src = np.zeros((1080, 1920, 3), np.uint8)
pend = cv.imread('mask_a2d_man.png')

x, y = np.mgrid[50:1080:150, 50:1920:150]
for position in zip(x, y):
    for p in zip(position[1], position[0]):
        p = np.float32(p)
        src = im_decorate(src, pend, p, 0.08, 90, 1)
# src = cv.bitwise_not(src)
cv.imshow('', src)
cv.waitKey()
