# coding: utf-8

import cv2 as cv
import numpy as np

from myutils import impreprocessing as impre
from myutils import imtemplate
from functools import partial
from myutils import ext_json
from myutils import imconverter as imcvt
import distort_model as net
from matplotlib import pyplot as plt


def find_four_corners(imsize, pnts):
    row, col = imsize

    top_left = (0, 0)
    top_right = (col, 0)
    bottom_left = (0, row)
    bottom_right = (col, row)

    def distance(base, pts):
        dist = np.sum(np.power(pts - base, 2), axis=1)
        return pnts[np.argmin(dist)]

    dist_partical = partial(distance, pts=pnts)
    return dist_partical(top_left), dist_partical(top_right), dist_partical(bottom_left), dist_partical(bottom_right)


def extractROI(white_img, calib_img, calib_shape, visibility=False):
    row, col = white_img.shape[:2]
    calib_r_num, calib_c_num = calib_shape

    # img = cv.cvtColor(white_img, cv.COLOR_BGR2GRAY)
    img = cv.fastNlMeansDenoising(white_img, 5, 7, 21)
    # img = cv.GaussianBlur(img, (5, 5), 0)

    kernel_dx = np.array([[3, 0, -3],
                          [3, 0, -3],
                          [3, 0, -3]])
    dx = cv.filter2D(img, cv.CV_32FC1, kernel_dx)
    dx = cv.convertScaleAbs(dx)
    _, dx = impre.otsu_threshold(dx, visibility=False)

    kernel_dy = np.array([[-3, -3, -3],
                          [0, 0, 0],
                          [3, 3, 3]])
    dy = cv.filter2D(img, cv.CV_32FC1, kernel_dy)
    dy = cv.convertScaleAbs(dy)
    _, dy = impre.otsu_threshold(dy, visibility=False)

    concat = cv.addWeighted(dx, 1, dy, 1, 0)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    concat = cv.morphologyEx(concat, cv.MORPH_DILATE, kernel=kernel, iterations=1)

    # 1. find external contour
    contours, _ = cv.findContours(concat, cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)
    # sort all of contours detected to
    contours = sorted(contours, key=lambda ct: cv.arcLength(ct, True), reverse=True)
    # attempt to extract relevantly exact contour
    cnt = contours[1]

    # 2. find min areaRect as target rectangle
    minRect = cv.minAreaRect(cnt)
    minrect_cx, minrect_cy = minRect[0]
    _rect_h, _rect_w = minRect[1]  # Note: h and w are both irregular, be careful
    _theta = minRect[2]  # range (0, 90), no negative, but positive angle
    angle = [_theta, _theta - 90][_rect_w > _rect_h]

    rot_mtx = cv.getRotationMatrix2D((minrect_cx, minrect_cy), angle, scale=1)

    minrect_h, minrect_w = sorted(minRect[1])

    boxcors = np.float32(cv.boxPoints(minRect))

    tl, tr, bl, br = find_four_corners((row, col), boxcors)

    tl = np.array(tl).T
    tl = np.insert(tl, 2, values=1, axis=0)
    tl = np.matmul(rot_mtx, tl).T

    grid_w = minrect_w / (calib_c_num - 1)
    grid_h = minrect_h / (calib_r_num - 1)

    tl_x = tl[0] - grid_w / 2
    tl_x = 0 if tl_x <= 0 else np.int0(tl_x)
    tl_y = tl[1] - grid_h / 2
    tl_y = 0 if tl_y <= 0 else np.int0(tl_y)

    br_x = tl[0] - grid_w / 2 + calib_c_num * grid_w
    br_x = col if br_x >= col else np.int0(br_x)
    br_y = tl[1] - grid_h / 2 + calib_r_num * grid_h
    br_y = row if br_y >= row else np.int0(br_y)

    # rotate image to horizontal
    rot_calib_img = cv.warpAffine(calib_img, rot_mtx, (col, row))
    roi_img = rot_calib_img[tl_y:br_y, tl_x:br_x]

    if visibility:
        tmp = rot_calib_img.copy()
        draw_line = partial(cv.line, tmp, color=0, thickness=3, lineType=cv.LINE_8)
        grid_w, grid_h = np.int0(grid_w), np.int0(grid_h)
        [draw_line((tl_x + c * grid_w, tl_y), (tl_x + c * grid_w, br_y)) for c in range(0, calib_c_num + 1)]
        [draw_line((tl_x, tl_y + r * grid_h), (br_x, tl_y + r * grid_h)) for r in range(0, calib_r_num + 1)]
        # cv.imwrite('rot_calib_mesh.png', tmp)
        tmp = cv.resize(tmp, None, fx=0.3, fy=0.3, interpolation=cv.INTER_AREA)
        cv.imshow('tmp', tmp)
        cv.waitKey()
        cv.destroyWindow('tmp')

    return roi_img


def grab_imgs():
    white_img = 0
    calib_img = 0
    return white_img, calib_img


def get_mid_grayscale(img):
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


def find_pnts_blob(bin_img, calib_shape, visibility=False):
    """
    This method is to extract points sorted by real position,
    :param bin_img:
    :param calib_shape:
    :param visibility:
    :return: return coordinate of points shape(row, col 2)
    """
    params = cv.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 5000
    params.minDistBetweenBlobs = 100
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.filterByConvexity = True
    params.minConvexity = 0.9
    detector = cv.SimpleBlobDetector_create(params)

    # Detect and visualize blobs
    # keypoints = detector.detect(bin_img)
    # im_with_keypoints = cv.drawKeypoints(bin_img, keypoints, np.array([]), (0, 0, 255),
    #                                       cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # print(len(keypoints))
    # im_with_keypoints = cv.resize(im_with_keypoints, None, fx=0.3, fy=0.3, interpolation=cv.INTER_AREA)
    # cv.imshow('keypoints', im_with_keypoints)
    # cv.waitKey()
    # cv.destroyWindow('keypoints')

    # method2 to find points
    flag = cv.CALIB_CB_SYMMETRIC_GRID
    # The function requires white space (like a square-thick border, the wider the better)
    # around the board to make the detection more robust in various environments.
    ret, pnts = cv.findCirclesGrid(bin_img, (calib_shape[1], calib_shape[0]), flags=flag, blobDetector=detector)
    if not ret:
        print(f'find points:{ret}')

    sorted_pnts = pnts.reshape(calib_shape[0], calib_shape[1], 2)
    if visibility:
        # draw chessboard corners
        temp = bin_img.copy()
        cv.drawChessboardCorners(temp, calib_shape, pnts, True)
        # cv.imwrite('chessboard_pnts.png', temp)
        temp = cv.resize(temp, None, fx=0.3, fy=0.3, interpolation=cv.INTER_AREA)
        cv.imshow('corners', temp)
        cv.waitKey()
        cv.destroyWindow('corners')
    return sorted_pnts


def extract_pnts_with_blob(roi, calib_shape, visibility=False):
    # Binarization
    MED_GRAYSCALE = get_mid_grayscale(roi)
    THRE_OFFSET = 35
    _, white = impre.otsu_threshold(roi, min_thre=MED_GRAYSCALE, offset=THRE_OFFSET, inv=True, visibility=visibility)
    _, black = impre.otsu_threshold(roi, max_thre=MED_GRAYSCALE, offset=THRE_OFFSET, visibility=visibility)

    # extract valid black and white points
    w_pnts = find_pnts_blob(white, calib_shape, visibility)
    b_pnts = find_pnts_blob(black, calib_shape, visibility)
    return w_pnts, b_pnts


def find_pnts_mincircle(cell, visibility):
    med_thre = get_mid_grayscale(cell)
    THRE_OFFSET = 5
    _, white = impre.otsu_threshold(cell, min_thre=med_thre, offset=THRE_OFFSET, visibility=False)
    _, black = impre.otsu_threshold(cell, max_thre=med_thre, offset=THRE_OFFSET, inv=True, visibility=False)

    def pnt_filter(bin_img):
        cnts, _ = cv.findContours(bin_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)
        centers = []
        area_ratio = 0.75

        if visibility:
            temp = cv.cvtColor(bin_img, cv.COLOR_GRAY2BGR)
            cv.drawContours(temp, cnts, -1, (0, 0, 255), 3)
            temp = imcvt.resize_for_display(temp)
            cv.imshow('cnts', temp)
            cv.waitKey(50)

        for cnt in cnts:
            area = cv.contourArea(cnt)
            print(area)
            center, radius = cv.minEnclosingCircle(cnt)
            if 220 < area < 1800 and area / np.pi / np.power(radius, 2) > area_ratio:
                centers.append(center)
        # TODO: have to detect one points
        assert len(centers) == 1, 'fail to detect points'
        return centers[0]

    w_pnt = pnt_filter(white)
    b_pnt = pnt_filter(black)

    if visibility:
        tmp = cell.copy()
        tmp = cv.cvtColor(tmp, cv.COLOR_GRAY2BGR)
        cv.circle(tmp, tuple(np.int0(w_pnt)), 2, (0, 0, 255), -1)
        cv.circle(tmp, tuple(np.int0(b_pnt)), 2, (0, 0, 255), -1)
        cv.imshow('cell', tmp)
        cv.waitKey(50)

    return w_pnt, b_pnt


def extract_pnts_with_mesh(roi_img, calib_shape, visibility=False):
    calib_r_num, calib_c_num = calib_shape
    roi_r, roi_c = roi_img.shape[:2]
    grid_w = roi_c // calib_c_num
    grid_h = roi_r // calib_r_num

    # adaptive padding to split an array into multiple sub-arrays of equal size
    c_pad = grid_w * calib_c_num - roi_c
    r_pad = grid_h * calib_r_num - roi_r
    roi_img_pad = roi_img.copy()
    if c_pad < 0:
        roi_img_pad = roi_img_pad[:, :c_pad]
    else:
        roi_img_pad = np.pad(roi_img_pad, ((0, 0), (0, c_pad)), 'constant', constant_values=((0, 0), (0, 0)))

    if r_pad < 0:
        roi_img_pad = roi_img_pad[:r_pad, :]
    else:
        roi_img_pad = np.pad(roi_img_pad, ((0, r_pad), (0, 0)), 'constant', constant_values=((0, 0), (0, 0)))

    cells = [np.hsplit(row, 16) for row in np.vsplit(roi_img_pad, 16)]
    all_pnts = [find_pnts_mincircle(cell, visibility) for row_img in cells for cell in row_img]

    w_pnts = []
    b_pnts = []
    for pnt in all_pnts:
        w_pnts.append(pnt[0])
        b_pnts.append(pnt[1])

    # top left point of each cell
    tl_x = [grid_w * c for c in range(calib_c_num)]
    tl_y = [grid_h * r for r in range(calib_r_num)]
    xv, yv = np.meshgrid(tl_y, tl_x, sparse=True)
    tl_pnts = [(x, y) for x in xv.flatten() for y in yv.flatten()]
    tl_pnts = np.asarray(tl_pnts)
    tl_pnts = tl_pnts[:, ::-1]

    #  absolute position of each points
    w_pnts = np.array(w_pnts) + tl_pnts
    b_pnts = np.array(b_pnts) + tl_pnts

    # visualization
    if visibility:
        tmp = roi_img.copy()
        draw_line = partial(cv.line, tmp, color=0, thickness=3, lineType=cv.LINE_8)
        [draw_line((c * grid_w, 0), (c * grid_w, roi_r)) for c in range(1, calib_c_num)]
        [draw_line((0, r * grid_h), (roi_c, r * grid_h)) for r in range(1, calib_r_num)]
        # cv.imwrite('roi_mesh.png', tmp)
        tmp = cv.cvtColor(tmp, cv.COLOR_GRAY2BGR)

        tl_pnts_tmp = tl_pnts.tolist()
        tl_pnts_tmp = [cv.KeyPoint(c[0], c[1], 1) for c in tl_pnts_tmp]
        tmp = cv.drawKeypoints(tmp, tl_pnts_tmp, None, (0, 255, 200))

        w_pnts_tmp = w_pnts.tolist()
        w_pnts_tmp = [cv.KeyPoint(c[0], c[1], 1) for c in w_pnts_tmp]
        tmp = cv.drawKeypoints(tmp, w_pnts_tmp, None, (0, 0, 255))

        b_pnts_tmp = b_pnts.tolist()
        b_pnts_tmp = [cv.KeyPoint(c[0], c[1], 1) for c in b_pnts_tmp]
        tmp = cv.drawKeypoints(tmp, b_pnts_tmp, None, (255, 255, 255))

        tmp = imcvt.resize_for_display(tmp)
        cv.imshow('tmp', tmp)
        cv.waitKey()
        cv.destroyWindow('tmp')

    w_pnts = w_pnts.reshape(calib_shape[0], calib_shape[1], 2)
    b_pnts = b_pnts.reshape(calib_shape[0], calib_shape[1], 2)

    return w_pnts, b_pnts


def decay_invalid_pnts(pnts_num, decay_factor):
    decay_pnts = pnts_num * decay_factor

    if np.ceil(decay_pnts) % 2 == 0:
        decay_pnts = np.ceil(decay_pnts)
    else:
        decay_pnts = np.floor(decay_pnts)

    if decay_pnts % 2:
        decay_pnts += 1

    res_pnts = pnts_num - decay_pnts

    return np.int0(res_pnts), np.int0(decay_pnts)


def get_roi_pnts(pnts, decay_factor=0.4):
    row, col = pnts.shape[:2]

    res_pnts_row, decay_pnts_row = decay_invalid_pnts(row, decay_factor)
    res_pnts_col, decay_pnts_col = decay_invalid_pnts(col, decay_factor)
    # print(f'{res_pnts_row=} | {res_pnts_col=}')

    roi_row_start = decay_pnts_row // 2
    roi_row_end = roi_row_start + res_pnts_row

    roi_col_start = decay_pnts_col // 2
    roi_col_end = roi_col_start + res_pnts_col

    return pnts[roi_row_start:roi_row_end, roi_col_start:roi_col_end]


def get_cvtcoef_pixel_mm(pnts, mm_gap):
    """
    The function is to get the coefficients of converting pixel to mm
    :param pnts: recommend using black points to compute cvtcoef
    :param mm_gap: the format is (row, col), unit is mm
    :return: return average cvtcoef
    """
    r_mm_space, c_mm_space = mm_gap
    roi = get_roi_pnts(pnts, decay_factor=0.6)
    row_pixel_space = np.diff(roi, axis=0)[:, :, 1]
    col_pixel_space = np.diff(roi, axis=1)[:, :, 0]
    cvtcoef = (np.mean(row_pixel_space) / r_mm_space + np.mean(col_pixel_space) / c_mm_space) / 2

    return cvtcoef


def transform_pnts(pnts):
    # moving black points based on white points
    roi_pnts = get_roi_pnts(pnts)

    # compute center position
    center = np.mean(roi_pnts, axis=(0, 1))

    # compute angles
    angle = calc_angle(roi_pnts)

    # rotate white and black points to Cartesian coordinates
    rot_mat = cv.getRotationMatrix2D(center.tolist(), angle, scale=1)
    # print(rot_mat_b, rot_mat_w)

    pnts = pnts.reshape(-1, 2).T
    pnts = np.insert(pnts, 2, values=1, axis=0)
    rot_pnts = np.matmul(rot_mat, pnts).T

    return rot_pnts, center, angle


def undist_using_nn(img, pnts1, pnts2):
    row, col = img.shape[:2]
    cx, cy = col / 2, row / 2
    pnts1 = pnts1 - np.array([cx, cy])
    pnts2 = pnts2 - np.array([cx, cy])

    model = net.training(pnts1.tolist(), pnts2.tolist())
    undist_img = net.undistorting(model, img)

    return undist_img


def undistorting(img, cvt_pnts_dev, flag=None):
    row, col = img.shape[:2]
    ipts_pnts = imtemplate.draw_calibboard((row, col), (16, 16))
    tgts_pnts = ipts_pnts - cvt_pnts_dev

    # ext_json.write('origin_pnts.json', ipts_pnts.tolist())
    # ext_json.write('undistorted_pnts.json', tgts_pnts.tolist())

    # using H matrix
    if flag == 'UNDIST_H':
        H, _ = cv.findHomography(ipts_pnts, tgts_pnts)
        img = cv.warpPerspective(img, H, (col, row), flags=cv.INTER_CUBIC)

    # use Neural network
    if flag == 'UNDIST_NN':
        img = undist_using_nn(img, tgts_pnts, ipts_pnts)

    cv.imwrite('calib_output.png', img)
    img = imcvt.resize_for_display(img)
    cv.imshow('calib', img)
    cv.waitKey()
    cv.destroyAllWindows()
    return img


def write_calib_data(pnts_deviation, cvtcoef):
    # write json
    cvt_pnts_dev = pnts_deviation / cvtcoef
    cvt_pnts_dev = cvt_pnts_dev.tolist()
    key = ['dx', 'dy']
    calib_data = dict()
    calib_data['map'] = [dict(zip(key, v)) for v in cvt_pnts_dev]
    calib_data['col'] = 16
    calib_data['row'] = 16
    calib_data['num'] = 256
    ext_json.write('calibrate.json', calib_data)


def calib_using_xml(img, path_mapx='mapx.xml', path_mapy='mapy.xml'):
    # read xml to remap
    fs_x = cv.FileStorage(path_mapx, flags=cv.FILE_STORAGE_FORMAT_XML)
    mapx = fs_x.getNode('mapx').mat()
    fs_y = cv.FileStorage(path_mapy, flags=cv.FILE_STORAGE_FORMAT_XML)
    mapy = fs_y.getNode('mapy').mat()
    undist_img = cv.remap(img, mapx, mapy, cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT, borderValue=0)
    cv.imwrite('undist_img.png', undist_img)


def fit_distcoff(img, src, tgt):
    row, col = img.shape[:2]
    cx, cy = col / 2, row / 2
    src = src - np.array([cx, cy])
    tgt = tgt - np.array([cx, cy])

    V = np.sum(src * src, axis=1)
    U = (tgt / src - 1) / np.vstack([V, V]).T

    U = np.hstack([U[:, 0], U[:, 1]])
    V = np.hstack([V, V])

    plt.scatter(V, U)
    plt.show()

    retval = cv.fitLine(np.array([np.vstack([V, U]).T]), cv.DIST_L12, 0, 0.01, 0.01)
    k2 = retval[1] / retval[0]
    k1 = retval[3] - k2 * retval[2]
    return -k1, -k2


def undistort_with_coff(img, src, tgt):
    k1, k2 = fit_distcoff(img, src, tgt)
    row, col = img.shape[:2]
    cx, cy = col / 2, row / 2
    src = src - np.array([cx, cy])
    V = np.sum(src * src, axis=1)
    coff = 1 + (k1 * V + k2 * V * V) * 0.5
    undist_pnts = src * np.vstack([coff, coff]).T + np.array([cx, cy])

    return undist_pnts


def show_res(roi, tr_b_pnts, tr_w_pnts, flag=None):
    canvas = np.zeros_like(roi)
    canvas.fill(110)

    # use H matrix
    if flag == 'UNDIST_H':
        H, _ = cv.findHomography(tr_w_pnts, tr_b_pnts)

        # perspective for points
        tr_w_pnts = cv.perspectiveTransform(np.array([tr_w_pnts]), H)
        tr_w_pnts = np.squeeze(tr_w_pnts)

        # undistort
        tr_w_pnts = undistort_with_coff(canvas, tr_w_pnts, tr_b_pnts)

    # use NN
    elif flag == 'UNDIST_NN':
        canvas = undist_using_nn(canvas, tr_b_pnts, tr_w_pnts)

    print(f'{flag=}')

    [cv.circle(canvas, (c[0], c[1]), 12, 255, -1) for c in np.int0(tr_w_pnts)]
    [cv.circle(canvas, (c[0], c[1]), 8, 0, -1) for c in np.int0(tr_b_pnts)]

    canvas = imcvt.resize_for_display(canvas)
    cv.imshow('canvas', canvas)
    cv.waitKey()
    cv.destroyWindow('canvas')


def calibrate():
    calib_shape = (16, 16)
    mm_gap = (4.68, 8.32)  # A2D(5.4, 9.6) 0.075  # Chair(4.68, 8.32) 0.065 -- pixel(72, 128)
    cvt_coef_mm_pixel = (124.8 + 0.5) / 1920  # A2D: 144.3 / 1920  # Chair: 124.8 / 1920
    print(f'{calib_shape=} | {mm_gap=} | {cvt_coef_mm_pixel=:.5f}')

    calib_img = cv.imread('calib_chair.bmp', cv.IMREAD_GRAYSCALE)
    white_img = cv.imread('calib_w.bmp', cv.IMREAD_GRAYSCALE)

    # extract region of interest, in order to draw mesh grid
    roi = extractROI(white_img, calib_img, calib_shape, visibility=False)

    # Method1: extract points using global threshold
    w_pnts, b_pnts = extract_pnts_with_blob(roi, calib_shape, False)

    # Method2: extract points using local threshold
    # w_pnts, b_pnts = extract_pnts_with_mesh(roi, calib_shape, False)

    # transform such as move and rotate black and white points in order to alignment
    tr_b_pnts, center_b, b_angle = transform_pnts(b_pnts)
    tr_w_pnts, center_w, w_angle = transform_pnts(w_pnts)
    center_bias = center_w - center_b
    tr_w_pnts -= center_bias

    # visualization
    show_res(roi, tr_b_pnts, tr_w_pnts, flag='UNDIST_H')

    # compute cvtcoef from pixel to mm
    cvtcoef = get_cvtcoef_pixel_mm(b_pnts, mm_gap)
    print(f'{cvtcoef=}')

    # compute the deviation between white and black points
    pnts_deviation = tr_w_pnts - tr_b_pnts
    cvt_pnts_dev = pnts_deviation / cvtcoef / cvt_coef_mm_pixel
    print(f'Max dev: {abs(cvt_pnts_dev).max()}')

    # write json
    # write_calib_data(pnts_deviation, cvtcoef)

    # undistort origin image to generate undistorted image
    img = cv.imread('./_data/ChairsideCalibrationFig1920_1080.png', cv.IMREAD_GRAYSCALE)
    undistorting(img, cvt_pnts_dev, flag='UNDIST_H')

    # for i in range(16):
    #     plt.plot(pnts_deviation[i * 16:(i + 1) * 16, 1])
    # plt.show()


if __name__ == '__main__':
    # method1: directly
    calibrate()

    # calib using xml
    # calib_img = cv.imread('./_data/A2D_calib.png', cv.IMREAD_GRAYSCALE)
    # calib_using_xml(calib_img)
