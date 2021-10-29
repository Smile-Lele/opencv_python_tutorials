# coding:utf-8

import random

import cv2 as cv
import numpy as np
from mypackage.multiplot import multiplot as mplt

COLORS = np.random.randint(0, 255, size=(100, 3)).tolist()


def otsu_threshold(img, min_thre=0, max_thre=255):
    img = cv.GaussianBlur(img, (5, 5), 0)

    img[img < min_thre] = min_thre
    img[img > max_thre] = max_thre

    thre, thre_img = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    print(f'otsu thre:{thre}')
    return thre, thre_img


def transform(src_img):
    imdict = dict()

    src_img_copy = src_img.copy()

    row, col, ch = src_img.shape
    center_x, center_y = (col // 2, row // 2)

    img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)

    img = cv.fastNlMeansDenoising(img, 5, 7, 21)
    img = cv.GaussianBlur(img, (5, 5), 0)

    # 1. find real corners as shape needed to transform
    sobelX = cv.Sobel(img, cv.CV_64F, 1, 0)
    sobelY = cv.Sobel(img, cv.CV_64F, 0, 1)

    sobelX = cv.convertScaleAbs(sobelX)  # np.uint8(np.absolute(sobelX))
    sobelY = cv.convertScaleAbs(sobelY)  # np.uint8(np.absolute(sobelY))

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

    _, sobelX = otsu_threshold(sobelX)
    thin_sobelX = cv.morphologyEx(sobelX, cv.MORPH_HITMISS, kernel=kernel, iterations=1)
    thin_sobelX = cv.bitwise_and(sobelX, cv.bitwise_not(thin_sobelX))

    _, sobelY = otsu_threshold(sobelY)
    thin_sobelY = cv.morphologyEx(sobelY, cv.MORPH_HITMISS, kernel=kernel, iterations=1)
    thin_sobelY = cv.bitwise_and(sobelY, cv.bitwise_not(thin_sobelY))

    imdict['thin_sobelX'] = thin_sobelX
    imdict['thin_sobelY'] = thin_sobelY

    sobel_ = cv.bitwise_or(thin_sobelX, thin_sobelY)
    imdict['sobel_'] = sobel_

    dilate_img = cv.morphologyEx(sobel_, cv.MORPH_DILATE, kernel=kernel, iterations=1)
    imdict['dilate'] = dilate_img

    # find external contour
    contours, _ = cv.findContours(dilate_img, cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)

    # sort all of contours detected to
    contours = sorted(contours, key=lambda ct: cv.arcLength(ct, True), reverse=True)

    cnt = contours[1]
    epsilon = 0.1 * cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)
    # print(approx.squeeze().tolist())
    assert len(approx) == 4, 'approx res should be 4'

    corners = approx.squeeze()
    for corner in corners:
        corner = tuple(corner)
        # cv.circle(src_img, corner, 6, random.choice(COLORS), -1)
        cv.drawMarker(src_img, corner, random.choice(COLORS), markerType=cv.MARKER_TILTED_CROSS, markerSize=30, thickness=3)
    imdict['src_img'] = src_img

    # 2. find min areaRect as target rectangle
    minRect = cv.minAreaRect(cnt)

    min_rect_center_x, min_rect_center_y = minRect[0]

    rect_height, rect_width = minRect[1]  # Note: h and w are both irregular, be careful

    _theta = minRect[2]  # range (0, 90), no negative, but positive angle

    angle = [_theta,  _theta - 90][rect_width > rect_height]

    min_rect_h, min_rect_w = sorted(minRect[1])

    roi_param_dict = dict()
    roi_param_dict['x'] = round(min_rect_center_x - min_rect_w / 2)
    roi_param_dict['y'] = round(min_rect_center_y - min_rect_h / 2)
    roi_param_dict['w'] = round(min_rect_w)
    roi_param_dict['h'] = round(min_rect_h)
    print(f'write ROI:{roi_param_dict}')

    # convert minRect to boxPoints
    box_cors = np.int0(cv.boxPoints(minRect))

    # draw minAreaRect on the source image
    cv.drawContours(src_img, [box_cors], 0, random.choice(COLORS), 3, cv.LINE_8)
    imdict['src_img'] = src_img


    def cart_to_polar(c):
        """
        sort corners by the distance between origin and four points
        :param c:
        :return: magnitude, angle
        """
        magnitude, angle_ = cv.cartToPolar(int(c[0]), int(c[1]), angleInDegrees=True)
        return magnitude[0], angle_[0]


    corners = sorted(list(corners), key=cart_to_polar)
    corners = np.float32(corners)
    box_cors = sorted(list(box_cors), key=cart_to_polar)
    box_cors = np.float32(box_cors)
    # print(corners)
    # print(box_cors)

    # perspective transform
    perspective_matrix = cv.getPerspectiveTransform(corners, box_cors, cv.DECOMP_LU)  # TODO: note DecompTypes
    perspective_im = cv.warpPerspective(src_img_copy, perspective_matrix, (col, row))
    # imgs_dict['perspective_im'] = perspective_im

    # rotate image to horizontal
    rotated_matrix = cv.getRotationMatrix2D((min_rect_center_x, min_rect_center_y), angle, scale=1)
    rotated_im = cv.warpAffine(perspective_im, rotated_matrix, (col, row))
    imdict['rotated_im'] = rotated_im


    # move image to the center of frame
    tx = center_x - min_rect_center_x
    ty = center_y - min_rect_center_y
    translation_matrix = np.matrix([[1, 0, tx],
                                    [0, 1, ty]], dtype=np.float32)
    translated_im = cv.warpAffine(rotated_im, translation_matrix, (col, row), cv.BORDER_TRANSPARENT)
    # imdict['translated_im'] = translated_im

    # mplot.show(imdict)

    return rotated_im
