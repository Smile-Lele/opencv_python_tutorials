# coding: utf-8

import cv2 as cv
import numpy as np
from functools import partial


def create_chessboard(mshape, pixels):
    """
    The function is to create a chessboard.
    :param mshape: (row, col)
    :param pixels: represents the pixel per grid
    :return: chessboard image
    """
    rows, cols = mshape
    height = (2 * rows - 1) * pixels
    width = (2 * cols - 1) * pixels
    img = np.zeros((rows, cols), np.uint8)

    # base_ = np.array([[0, 1], [1, 0]])
    # img = np.tile(base_, (4, 4))

    img[1::2, ::2] = 255
    img[::2, 1::2] = 255
    img = cv.resize(img, (width, height), interpolation=cv.INTER_AREA)
    info = 'chessboard: \nmshape:{} \npixels:{}\n'
    info = info.format(mshape, pixels)
    print(info)
    return img


def draw_mesh(image, mshape):
    """
    This is function to draw mesh (row, col) on the image
    Note: copy is shadow copy in order to block
    :param image: any
    :param mshape: it is a tuple(row, col)
    :return: meshed image
    """
    img = image.copy()
    rows, cols = mshape
    im_r, im_c = img.shape[:2]

    r_grid = im_r // rows
    c_grid = im_c // cols

    draw_line = partial(cv.line, img, color=0, thickness=3, lineType=cv.LINE_8)

    [draw_line((c * c_grid, 0), (c * c_grid, im_r)) for c in range(1, cols)]
    [draw_line((0, r * r_grid), (im_c, r * r_grid)) for r in range(1, rows)]

    info = 'mesh: \nimshape:{} \nmshape:{}\n'
    info = info.format(image.shape, mshape)
    print(info)
    return img


def create_canvas(dsize):
    """
    The function creates canvas with dsize, grayscale per pixel is 0
    :param dsize: img(row, col)
    :return: empty or black image
    """
    canvas = np.zeros(dsize, np.uint8)
    info = 'canvas: \nshape:{} \ndtype:{}\n'
    info = info.format(canvas.shape, canvas.dtype)
    print(info)
    return canvas


def create_whiteboard(dsize):
    """
    The function creates white board with dsize, grayscale per pixel is 255
    :param dsize: image(row, col)
    :return: white image
    """
    canvas = np.zeros(dsize, np.uint8)
    canvas.fill(255)
    info = 'whiteboard: \nshape:{} \ndtype:{}\n'
    info = info.format(canvas.shape, canvas.dtype)
    print(info)
    return canvas


def draw_gradient(dsize, gap: int):
    grad_v_mat, grad_h_mat = np.mgrid[0:256:gap, 0:256:gap].astype(np.uint8)
    grad_h = cv.resize(grad_h_mat, tuple(reversed(dsize)), interpolation=cv.INTER_AREA)
    grad_v = cv.resize(grad_v_mat, tuple(reversed(dsize)), interpolation=cv.INTER_AREA)
    info = 'gradient: \nshape:{} \ndtype:{}\n'
    info = info.format(grad_h.shape, grad_h.dtype)
    print(info)
    return grad_h, grad_v