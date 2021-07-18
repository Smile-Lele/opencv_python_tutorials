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


def mat_to_img(mat, dsize):
    """
    The function is to convert matrix to image, the grayscale per pixel is
    from the value of matrix corresponding to the region. To be specific,
    each pixel in a region has same grayscale.
    :param mat: matrix needed to be convert
    :param dsize: image shape (row, col)
    :return: image(row, col)
    """
    img = np.zeros(dsize, dtype=np.uint8)
    print(f'mat:({mat.shape},{mat.dtype}) -> src:({img.shape},{img.dtype})')

    # TODO: whether scale up or down has diff strategies of interpolation
    img = cv.resize(mat, (dsize[1], dsize[0]), interpolation=cv.INTER_CUBIC)
    return img


def scale_mat(mat):
    """
    The function is to scale up and down source matrix to maximum grayscale 250
    :param mat: input matrix
    :return: matrix processed
    """
    scale = 250 / mat.max()
    new_mat = np.round(np.multiply(mat, scale, dtype=np.float16)).astype(dtype=np.uint8)
    return new_mat


def masking(src:np.uint8, mask:np.uint8):
    """
    TODO: write description
    :param mask:
    :param src:
    :return:
    """
    scale = np.divide(mask, 255)
    image = np.round(np.multiply(src, scale, dtype=np.float16)).astype(dtype=np.uint8)
    image = scale_mat(image)
    return image


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
