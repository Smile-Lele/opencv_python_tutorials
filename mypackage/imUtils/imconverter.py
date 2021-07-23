# coding: utf-8

import numpy as np
import cv2 as cv


def img_to_mat(img, mshape):
    """
    The function will convert image to matrix, each region of which is
    from the average of grayscale in the whole region.
    :param img: image should be a single channel matrix
    :param mshape: it is a tuple (row, col)
    :return: matrix(row, col)
    """
    imrows, imcols = img.shape[:2]
    matrows, matcols = mshape

    mat = np.zeros(mshape, dtype=img.dtype)

    print(f'src:({img.shape},{img.dtype}) -> mat:({mat.shape},{mat.dtype})')


    r_step = imrows // matrows
    c_step = imcols // matcols

    for r in range(matrows):
        r_start = int(r * r_step)
        r_end = r_start + r_step
        r_end = [r_end, imrows][r_end > imrows]

        for c in range(matcols):
            c_start = int(c * c_step)
            c_end = c_start + c_step
            c_end = [c_end, imcols][c_end > imcols]

            _avg = img[r_start:r_end, c_start:c_end]
            mat[r][c] = np.average(_avg)
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
