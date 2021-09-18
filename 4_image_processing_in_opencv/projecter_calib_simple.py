# coding: utf-8

from functools import partial

import cv2 as cv
import numpy as np


def extractROI():
    # project white

    # save ROI to json

    pass


def grab_imgs():
    imgs = 0
    return imgs


def preprocess(imgs):
    # concat

    # denoise

    # load ROI

    roi_img = imgs[:, :]
    return roi_img


def get_mid_grayscale(img):
    # img = cv.fastNlMeansDenoising(img, None, 5, 7, 21)
    # img = cv.GaussianBlur(img, (5, 5), 0)

    hist = cv.calcHist([img], [0], mask=None, histSize=[256], ranges=[0, 256])
    hist = hist.squeeze()
    hist_list = hist.tolist()
    max_grayscale = hist_list.index(max(hist_list))
    return max_grayscale


def calc_angle(pnts):
    row, col = pnts.shape[:2]

    if row >= col:
        lines = [cv.fitLine(pnts[:, c], cv.DIST_WELSCH, 0, 0.01, 0.01) for c in range(col)]
        mean_slope = np.mean([line[0] / line[1] for line in lines])
        angle = -np.degrees(np.arctan(mean_slope))
    else:
        lines = [cv.fitLine(pnts[r, :], cv.DIST_WELSCH, 0, 0.01, 0.01) for r in range(row)]
        mean_slope = np.mean([line[0] / line[1] for line in lines])
        angle = (90 - np.degrees(np.arctan(mean_slope)))

    return angle


def extract_pnts(bin_img, calib_shape, visibility=False):
    # method2 to find points
    ret, pnts = cv.findCirclesGrid(bin_img, calib_shape)
    print(f'find points:{ret}')

    # fast method to find sub pixel
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    cv.cornerSubPix(bin_img, pnts, calib_shape, (-1, -1), criteria)

    # accurate method to find sub pixel
    # cv.find4QuadCornerSubpix(bin_img, pnts, calib_shape)

    sorted_pnts = pnts.reshape(calib_shape[0], calib_shape[1], 2)
    if visibility:
        # draw chessboard corners
        temp = bin_img.copy()
        cv.drawChessboardCorners(temp, calib_shape, pnts, True)
        temp = cv.resize(temp, None, fx=0.3, fy=0.3, interpolation=cv.INTER_AREA)
        cv.imshow('corners', temp)
        cv.waitKey()
        cv.destroyWindow('corners')
    return sorted_pnts


def decay_invalid_pnts(pnts_num, decay_factor=0.4):
    decay_pnts = pnts_num * decay_factor

    if np.ceil(decay_pnts) % 2 == 0:
        decay_pnts = np.ceil(decay_pnts)
    else:
        decay_pnts = np.floor(decay_pnts)

    if decay_pnts % 2:
        decay_pnts += 1

    res_pnts = pnts_num - decay_pnts
    # print(np.int0(res_pnts), np.int0(decay_pnts))
    return np.int0(res_pnts), np.int0(decay_pnts)


def get_roi_pnts(pnts):
    row, col = pnts.shape[:2]

    res_pnts_row, decay_pnts_row = decay_invalid_pnts(row)
    res_pnts_col, decay_pnts_col = decay_invalid_pnts(col)

    roi_row_start = decay_pnts_row // 2
    roi_row_end = roi_row_start + res_pnts_row

    roi_col_start = decay_pnts_col // 2
    roi_col_end = roi_col_start + res_pnts_col
    # print(roi_row_start, roi_row_end, roi_col_start, roi_col_end)
    return pnts[roi_row_start:roi_row_end, roi_col_start:roi_col_end]


def get_pnts_deviation(b_pnts, w_pnts):
    # moving black points based on white points
    roi_pnts_b = get_roi_pnts(b_pnts)
    roi_pnts_w = get_roi_pnts(w_pnts)

    mean_center_b = np.mean(roi_pnts_b, axis=(0, 1))
    mean_center_w = np.mean(roi_pnts_w, axis=(0, 1))

    center_diff = mean_center_w - mean_center_b
    print(f'dx:{center_diff[0]:.2f}, dy:{center_diff[1]:.2f}')

    b_angle = calc_angle(roi_pnts_b)
    w_angle = calc_angle(roi_pnts_w)
    print(f'Angle_b:{b_angle:.3f}, Angle_w:{w_angle:.3f}')

    # rotate white and black points to Cartesian coordinates
    rot_mat_b = cv.getRotationMatrix2D(mean_center_b.tolist(), b_angle, scale=1)[:, :2]
    rot_mat_w = cv.getRotationMatrix2D(mean_center_w.tolist(), w_angle, scale=1)[:, :2]
    # print(rot_mat_b, rot_mat_w)

    b_pnts = np.matmul(b_pnts.reshape(-1, 2), rot_mat_b)
    w_pnts = np.matmul(w_pnts.reshape(-1, 2), rot_mat_w)

    pnts_deviation = b_pnts - w_pnts + center_diff

    # return pnts_deviation
    return mean_center_b, mean_center_w, b_angle, w_angle


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


def main():
    calib_shape = (16, 16)
    src = cv.imread('_data/aaabbbccc.jpg', cv.IMREAD_GRAYSCALE)
    roi = preprocess(src)

    # binarization
    MAX_GRAYSCALE = get_mid_grayscale(roi)
    THRE_OFFSET = 45
    _, white = otsu_threshold(roi, min_thre=MAX_GRAYSCALE + THRE_OFFSET)
    white = cv.bitwise_not(white)

    _, black = otsu_threshold(roi, max_thre=MAX_GRAYSCALE - THRE_OFFSET)
    # black = cv.bitwise_not(black)

    # visualization: show thre_img
    # temp = np.hstack((roi, white, black))
    # temp = cv.resize(temp, None, fx=0.2, fy=0.2, interpolation=cv.INTER_AREA)
    # cv.imshow('', temp)
    # cv.waitKey()

    w_pnts = extract_pnts(white, calib_shape)
    b_pnts = extract_pnts(black, calib_shape)

    # pnts_deviation = get_pnts_deviation(b_pnts, w_pnts)
    # print(np.round(pnts_deviation.reshape(calib_shape), 3))

    # visualization: show result
    row, col = roi.shape[:2]
    mean_center_b, mean_center_w, b_angle, w_angle = get_pnts_deviation(b_pnts, w_pnts)
    rotated_matrix = cv.getRotationMatrix2D(mean_center_b.tolist(), b_angle, scale=1)
    rotated_im_b = cv.warpAffine(black, rotated_matrix, (col, row))
    cv.drawMarker(rotated_im_b, np.int32(mean_center_b.tolist()), 0, markerType=cv.MARKER_DIAMOND, markerSize=50, thickness=5)

    rotated_matrix = cv.getRotationMatrix2D(mean_center_w.tolist(), w_angle, scale=1)
    rotated_im_w = cv.warpAffine(white, rotated_matrix, (col, row))
    cv.drawMarker(rotated_im_w, np.int32(mean_center_w.tolist()), 0, markerType=cv.MARKER_DIAMOND, markerSize=50, thickness=5)

    # move image to the center of frame

    tx, ty = -(mean_center_w - mean_center_b)
    translation_matrix = np.matrix([[1, 0, tx],
                                    [0, 1, ty]], dtype=np.float32)
    translated_im = cv.warpAffine(rotated_im_w, translation_matrix, (col, row), cv.BORDER_TRANSPARENT)

    add = cv.addWeighted(cv.bitwise_not(rotated_im_b), 1, cv.bitwise_not(translated_im), 1, 0)
    temp = np.hstack((roi, add))
    temp = cv.resize(temp, None, fx=0.2, fy=0.2, interpolation=cv.INTER_AREA)
    cv.imshow('', temp)
    cv.waitKey()


if __name__ == '__main__':
    main()
