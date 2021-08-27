# coding: utf-8

import cv2 as cv
import numpy as np


def img_to_mat(img, mshape):
    """
    The function will convert image to matrix, each region of which is
    from the average of grayscale in the whole region.
    :param img: image should be a single channel matrix
    :param mshape: it is a tuple (row, col)
    :return: matrix(row, col)
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    imrows, imcols = img.shape[:2]
    matrows, matcols = mshape
    r_step = imrows // matrows
    c_step = imcols // matcols
    ceil_counter = np.ones((imrows, imcols))
    ceil_add = np.add.reduceat(np.add.reduceat(img, np.arange(0, img.shape[0], r_step), axis=0),
                               np.arange(0, img.shape[1], c_step), axis=1)
    ceil_nums = np.add.reduceat(np.add.reduceat(ceil_counter, np.arange(0, ceil_counter.shape[0], r_step), axis=0),
                                np.arange(0, ceil_counter.shape[1], c_step), axis=1)
    mat = np.divide(ceil_add, ceil_nums).astype(np.uint8)

    print(f'src:({img.shape},{img.dtype}) -> mat:({mat.shape},{mat.dtype})')
    return mat


def resize_ex(src: (np.uint8, np.float32), dsize):
    """
    This is a extension of resize
    TODO: need to be verified
    :param src:
    :param dsize:
    :return:
    """
    print(f'resize: \nsrc{src.shape} -> dst:{dsize}')
    inter_type = [cv.INTER_CUBIC, cv.INTER_AREA][src.size > np.cumprod(dsize).max()]
    dst = cv.resize(src, tuple(reversed(dsize)), interpolation=inter_type)
    return dst


def gen_mask(src):
    """
    :param src:
    :return:
    """
    # create mask
    mask = np.zeros(src.shape, dtype=np.float16)
    mask.fill(255)

    min_gray = np.min(src)
    scales = np.divide(src, min_gray)
    mask = np.round(np.divide(mask, scales)).astype(dtype=np.uint8)
    return mask


def masking(src: np.uint8, mask: np.uint8):
    """
    TODO: write description
    :param mask:
    :param src:
    :return:
    """
    scales = np.divide(mask, 255)
    image = np.round(np.multiply(src, scales, dtype=np.float16)).astype(dtype=np.uint8)
    image = scaleAbs_ex(image, 255)
    return image


def scaleAbs_ex(src, maxVal):
    dst = cv.convertScaleAbs(src, alpha=maxVal / src.max())
    return dst


def evaluating(mat):
    """
    TODO: write description
    :param mat:
    :return:
    """
    max_ = mat.max()
    min_ = mat.min()
    # namean_ = np.average(mat)
    range_ = max_ - min_
    mean, stddev = cv.meanStdDev(mat)

    print(f'evaluate: \nmean:{mean}, stddev:{stddev}, range:{range_}')

    return range_
