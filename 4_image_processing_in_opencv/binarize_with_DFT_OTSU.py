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


def detect_edge_with_dft(img, radius):
    rows, cols = img.shape
    print(img.shape)

    nrows = cv.getOptimalDFTSize(rows)
    ncols = cv.getOptimalDFTSize(cols)

    right = ncols - cols
    bottom = nrows - rows
    nimg = cv.copyMakeBorder(img, 0, bottom, 0, right, cv.BORDER_CONSTANT, value=0)
    print(nimg.shape)

    dft = cv.dft(np.float32(nimg), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    dft_ch1, dft_ch2 = cv.split(dft_shift)
    magnitude_spectrum = 20 * np.log(cv.magnitude(dft_ch1, dft_ch2))
    imdict['magnitude_spectrum'] = magnitude_spectrum

    c_r = nrows // 2
    c_c = ncols // 2

    mask = np.ones((nrows, ncols, 2), dtype=nimg.dtype)
    mask = cv.circle(mask, (c_c, c_r), radius, (0, 0), -1, cv.LINE_8)

    # apply mask and inverse DFT
    fshift = dft_shift * mask
    fshift[fshift == 0] = 0.001
    f_ch1, f_ch2 = cv.split(fshift)
    magnitude_spectrum_masked = 20 * np.log(cv.magnitude(f_ch1, f_ch2))
    imdict['magnitude_spectrum_masked'] = magnitude_spectrum_masked

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv.idft(f_ishift)
    img_back_ch1, img_back_ch2 = cv.split(img_back)
    img_back = cv.magnitude(img_back_ch1, img_back_ch2)
    img_back = cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX)

    return img_back.astype(np.uint8)


if __name__ == '__main__':
    imdict = dict()
    img = cv.imread('../mydata/captured_white.png', 0)
    img = cv.fastNlMeansDenoising(img, None, 5, 7, 21)
    imdict['img'] = img

    img_back = detect_edge_with_dft(img, 80)
    imdict['img_back'] = img_back

    # _, img_th = otsu_threshold(img_back)
    # imdict['img_th'] = img_th
    #
    # kernel = cv.getStructuringElement(cv.MORPH_CROSS, (5,5))
    # dilate_img = cv.dilate(img_th, kernel=kernel, iterations=2)
    # imdict['dilate'] = dilate_img

    # img = cv.GaussianBlur(img, (5, 5), 0)
    canny = cv.Canny(img_back, 180, 250)
    imdict['canny'] = canny

    # cv.CV_64F is order to avoid negative num
    sobelX = cv.Sobel(img_back, cv.CV_64F, 1, 0)
    sobelY = cv.Sobel(img_back, cv.CV_64F, 0, 1)

    sobelX = cv.convertScaleAbs(sobelX)  # np.uint8(np.absolute(sobelX))
    sobelY = cv.convertScaleAbs(sobelY)  # np.uint8(np.absolute(sobelY))

    start_t = time.time()
    _, sobelX = otsu_threshold(sobelX)
    _, sobelY = otsu_threshold(sobelY)
    print(time.time() - start_t)

    sobel_ = cv.bitwise_or(sobelX, sobelY)

    imdict['sobelX'] = sobelX
    imdict['sobelY'] = sobelY
    imdict['sobel_'] = sobel_

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dilate_img = cv.morphologyEx(sobel_, cv.MORPH_DILATE, kernel=kernel, iterations=2)
    dilate_img = cv.morphologyEx(dilate_img, cv.MORPH_CLOSE, kernel=kernel, iterations=2)
    imdict['dilate'] = dilate_img

    cnts, hier = cv.findContours(dilate_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnt = cnts[0]
    epsilon = 0.1 * cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)
    print(approx.squeeze().tolist())
    assert len(approx) == 4, 'approx res should be 4'

    mplt.show(imdict)