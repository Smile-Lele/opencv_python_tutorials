# coding:utf-8
import os
from concurrent import futures
from functools import partial

import cv2 as cv
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from scipy import signal


def otsu_threshold(img, min_thre=0, max_thre=255, offset=0, inv=False, visibility=False):
    img = cv.GaussianBlur(img, (3, 3), 0)

    min_thre += offset
    max_thre -= offset
    img[img < min_thre] = min_thre
    img[img > max_thre] = max_thre

    thre, thre_img = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print(f'otsu thre:{thre}')

    if inv:
        thre_img = cv.bitwise_not(thre_img)

    if visibility:
        temp = cv.resize(thre_img, None, fx=0.3, fy=0.3, interpolation=cv.INTER_AREA)
        cv.imshow('otsu_thre', temp)
        cv.waitKey()
        cv.destroyWindow('otsu_thre')
    return thre, thre_img


def de_noise(image, blurflag=True):
    """
    The function is to remove noise and smooth edges
    :param image: source image
    :param blurflag: default blurflag is True
    :return: processed image
    """
    image = image.copy()
    image = cv.fastNlMeansDenoising(image, 5, 7, 21)
    if blurflag:
        image = cv.GaussianBlur(image, (5, 5), 0)
    return image


def concat(imgs, undistort_flag=False):
    """
    concat all of images captured by camera
    in order to remove noise
    :param undistort_flag: un-distort flag, default is False
    :param imgs: images need to be processed
    :return: img
    """
    global undistort
    if undistort_flag:
        # read the parameter of camera calibration
        mtx, dist = ext_json.read_cam_para()
        mtx = np.array(mtx)
        dist = np.array(dist)
        undistort = partial(undistorting, mtx=mtx, dist=dist)

    _sum = 0
    for im in imgs:
        if undistort_flag:
            im = undistort(im)
        _sum = np.add(_sum, im, dtype=np.int64)
    img = np.around(np.divide(_sum, len(imgs))).astype(dtype=np.uint8)
    img = de_noise(img, blurflag=False)
    return img


def divide_images_helper(index_, imgs, mshape):
    """
    The function is concat a batch of images and cut it as average grids
    :param index_: in order to multiple threads
    :param imgs: a batch of images
    :param mshape: (row, col)
    :return: tuple(index, matrix)
    """
    img = concat(imgs, undistort_flag=True)
    temp_path = os.path.join('./_data', 'img')
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    cv.imwrite(os.path.join(temp_path, str(index_) + '.png'), img)
    mat = imcvt.img_to_mat(img, mshape)
    return index_, mat


def divide_images(grab_imgs, mshape):
    """
    The function is same to the above-mentioned function cut,
    but it is multiple threads to process data
    :param grab_imgs: all of images captured from camera per intensity
    :param mshape: mesh(row, col)
    :return: a list of matrix averaged on mesh
    """
    to_do_list = list()
    MAX_WORKERS = len(grab_imgs)
    with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for index_, imgs in enumerate(grab_imgs):
            future = executor.submit(divide_images_helper, index_, imgs, mshape)
            to_do_list.append(future)
        done_iter = futures.as_completed(to_do_list)
        done_iter = tqdm.tqdm(done_iter, total=len(grab_imgs), desc='Ceil images')

        res = [future.result() for future in done_iter]
        sort_res = sorted(res, key=lambda r: r[0])
        avg_ims = [res[1] for res in sort_res]
    return avg_ims


def fit_data(scales, avg_ims):
    print('fit data')
    row, col = avg_ims[0].shape[:2]
    func_dict = dict()
    for r in range(row):
        for c in range(col):
            grayscales = [img[r, c] for img in avg_ims]
            ply_fit = np.polyfit(scales, grayscales, 1)
            func_dict['_'.join(['f', str(r), str(c)])] = ply_fit.tolist()
            f = np.poly1d(ply_fit, variable='x')

            plt.plot(scales, f(scales), color='blue')
            plt.plot(scales, grayscales, color='red')

    print('write json')
    ext_json.write('./_data/func_data.json', func_dict)
    plt.show()


def is_grayscale_available(image):
    grayscale_range = range(190, 220)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    hist = cv.calcHist([image], [0], None, [255], [-10, 260])
    hist_list = list(hist.T[0])
    # plt.plot(hist_list)
    # plt.show()

    # get the grayscale corresponding to the valley between two peaks
    peaks, _ = signal.find_peaks(hist_list, distance=30)
    peaks = sorted(peaks, key=lambda x: hist_list[x], reverse=True)
    two_peaks = peaks[:2]
    bright_peak = max(two_peaks)

    info = '{} -> peak:{}'
    print(info.format(peaks, bright_peak))

    if bright_peak < min(list(grayscale_range)):
        return -1
    elif bright_peak > max(list(grayscale_range)):
        return 1

    return 0