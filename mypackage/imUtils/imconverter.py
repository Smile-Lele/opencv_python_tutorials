# coding: utf-8

import cv2 as cv
import numpy as np


def imread_ex(filename, flags):
    return cv.imdecode(np.fromfile(filename, dtype=np.uint8), flags)


def img_to_mat(img, mshape):
    """
    The function will convert image to matrix, each region of which is
    from the average of grayscale in the whole region.
    :param img: image should be a single channel matrix
    :param mshape: it is a tuple (row, col)
    :return: matrix(row, col)
    """
    if img is not None and len(img.shape) == 3 and img.depth == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    mrows, mcols = mshape
    ceils = [np.array_split(row, mcols, axis=1) for row in np.array_split(img, mrows, axis=0)]
    means = [np.mean(ceil) for row_img in ceils for ceil in row_img]
    mat = np.array(means).reshape(mshape)

    # print(f'src:({img.shape},{img.dtype}) -> mat:({mat.shape},{mat.dtype})')
    return mat


def resize_ex(src: (np.uint8, np.float32), dsize):
    """
    This is a extension of resize
    TODO: need to be verified
    :param src:
    :param dsize:
    :return:
    """
    # print(f'resize: \nsrc{src.shape} -> dst:{dsize}')
    inter_type = [cv.INTER_CUBIC, cv.INTER_AREA][src.size > dsize[0] * dsize[1]]
    dst = cv.resize(src, tuple(reversed(dsize)), interpolation=inter_type)
    return dst


def resize_for_display(img, win_size=(1920 * 4 / 5, 1080 * 4 / 5)):
    """
    The method is to make image to fit window size in the win while displaying
    :param img:
    :param win_size:
    :return:
    """
    row, col = img.shape[:2]
    win_c, win_r = win_size
    row_ratio = win_r / row
    col_ratio = win_c / col
    scale_ratio = row_ratio if row_ratio <= col_ratio else col_ratio
    if scale_ratio == 1:
        return img
    else:
        INTERPOLATION = cv.INTER_AREA if scale_ratio < 1 else cv.INTER_CUBIC
        img = cv.resize(img, None, fx=scale_ratio, fy=scale_ratio, interpolation=INTERPOLATION)
        return img


def img_to_mask(src):
    """
    :param src:
    :return:
    """
    # create mask
    mask = np.zeros_like(src, dtype=np.float32)
    mask.fill(255)
    min_ = src.min() if src.min() != 0 else src.min() + 1
    mask = mask / (src / min_)
    return mask


def bitwise_mask(src, mask):
    """
    The method is to adjust the grayscale of each pixels in source image.
    :param mask:
    :param src:
    :return:
    """
    if mask.size < src.size:
        mask = cv.resize(mask, (src.shape[1], src.shape[0]), interpolation=cv.INTER_AREA)

    masked_img = cv.convertScaleAbs(src * (mask / 255))
    return masked_img


def scaleAbs_ex(src, maxVal):
    dst = cv.convertScaleAbs(src, alpha=maxVal / src.max())
    return dst


def remap_ex(img, mapx, mapy):
    row, col = img.shape[:2]
    mapx = mapx.reshape(row, col).astype(np.float32)
    mapy = mapy.reshape(row, col).astype(np.float32)
    new_img = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    return new_img


def cvtColor_ex(img):
    if len(img.shape) == 3 and len(img.shape[-1] == 3):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img
