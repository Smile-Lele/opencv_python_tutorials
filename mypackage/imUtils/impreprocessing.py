# coding:utf-8
import glob
import os
import random
from concurrent import futures
from functools import partial

import cv2 as cv
import numpy as np
import tqdm
from scipy import signal

from calib_lib.cam_undistort import undistorting
from myutils import ext_json
from myutils import imconverter as imcvt


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
        image = cv.edgePreservingFilter(image, sigma_s=25, sigma_r=0.3, flags=cv.RECURS_FILTER)
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


def divide_images_helper(index_, img, mshape):
    """
    The function is concat a batch of images and cut it as average grids
    :param index_: in order to multiple threads
    :param imgs: a batch of images
    :param mshape: (row, col)
    :return: tuple(index, matrix)
    """
    # img = concat(imgs, undistort_flag=True)
    # temp_path = os.path.join('./_data', 'img')
    # if not os.path.exists(temp_path):
    #     os.makedirs(temp_path)
    # cv.imwrite(os.path.join(temp_path, str(index_) + '.png'), img)
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
        for index_, img in enumerate(grab_imgs):
            future = executor.submit(divide_images_helper, index_, img, mshape)
            to_do_list.append(future)
        done_iter = futures.as_completed(to_do_list)
        done_iter = tqdm.tqdm(done_iter, total=len(grab_imgs), desc='Ceil images')

        res = [future.result() for future in done_iter]
        sort_res = sorted(res, key=lambda r: r[0])
        avg_ims = [res[1] for res in sort_res]

    return avg_ims


def is_grayscale_available(image):
    grayscale_range = range(190, 230)
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


def remove_anomaly_gaussian(imgs):
    """
    This function is to remove less anomaly images
    :param imgs:
    :return: get images in larger possibility distribution
    """
    im_gray_mean = [np.mean(img) for img in imgs]
    min_, max_ = np.argmin(im_gray_mean), np.argmax(im_gray_mean)
    imgs.pop(min_)
    imgs.pop(max_)

    im_gray_mean = [np.mean(img) for img in imgs]
    im_gray_mean = np.asarray(im_gray_mean)
    mean, std = im_gray_mean.mean(), im_gray_mean.std()
    cond = np.abs(im_gray_mean - mean) < std
    valid_index = np.where(cond)
    # print(valid_index[0].tolist())
    valid_imgs = [imgs[i] for i in valid_index[0].tolist()]
    print(f'total: {len(imgs)}, valid: {len(valid_imgs)}')

    return valid_imgs


def remove_anomaly_kmeans(imgs):
    data = [np.mean(img) for img in imgs]

    data = np.asarray(data, dtype=np.float32).reshape(-1, 1)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-3)
    flags = cv.KMEANS_RANDOM_CENTERS

    # 应用K均值
    compactness, labels, centers = cv.kmeans(data, 5, None, criteria, 100, flags)
    # print(centers)

    class_index = np.argsort(centers, axis=0)
    med_class_index = class_index[1:4].ravel()

    valid_region = np.zeros_like(labels, bool)
    for i in med_class_index:
        valid_region |= (labels == i)

    valid_index = np.where(valid_region)
    valid_imgs = [imgs[i] for i in valid_index[0].tolist()]
    print(f'total: {len(imgs)}, valid: {len(valid_imgs)}')
    return valid_imgs


def remove_anomaly_ransac(imgs, sigma):
    data = [np.mean(img) for img in imgs]

    d_min, d_max = np.min(data), np.max(data)
    pivot = np.linspace(d_min, d_max + 1, num=np.int0(np.ptp(data)))
    max_pnts_num = -1
    retpivot = 0
    for p in pivot:
        region_pnts_num = len(data) - (np.sum(data > p + sigma) + np.sum(data < p - sigma))
        if region_pnts_num >= max_pnts_num:
            max_pnts_num = region_pnts_num
            retpivot = p
        if max_pnts_num == -1:
            print(p)

    min_index = np.argwhere(np.asarray(data) >= retpivot - sigma).ravel()
    max_index = np.argwhere(np.asarray(data) <= retpivot + sigma).ravel()
    union_index = set(min_index) & set(max_index)

    valid_imgs = [imgs[i] for i in union_index]
    print(f'total: {len(imgs)}, valid: {len(valid_imgs)}')

    return valid_imgs
