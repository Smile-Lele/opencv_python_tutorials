# coding: utf-8

import cv2 as cv
import numpy as np

from mypackage.imUtils import impreprocessing as impre
from mypackage.imUtils import imtemplate
from functools import partial
from mypackage.fileUtils import read_write_json


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


def extract_pnts_with_circle(roi, calib_shape, visibility=False):
    # Binarization
    MED_GRAYSCALE = get_mid_grayscale(roi)
    THRE_OFFSET = 35
    _, white = impre.otsu_threshold(roi, min_thre=MED_GRAYSCALE, offset=THRE_OFFSET, inv=True, visibility=visibility)
    _, black = impre.otsu_threshold(roi, max_thre=MED_GRAYSCALE, offset=THRE_OFFSET, visibility=visibility)

    # extract valid black and white points
    w_pnts = find_pnts_blob(white, calib_shape, visibility)
    b_pnts = find_pnts_blob(black, calib_shape, visibility)
    return w_pnts, b_pnts, white, black


def find_pnts_mincircle(cell, visibility):
    med_thre = get_mid_grayscale(cell)
    THRE_OFFSET = 15
    _, white = impre.otsu_threshold(cell, min_thre=med_thre, offset=THRE_OFFSET, visibility=False)
    _, black = impre.otsu_threshold(cell, max_thre=med_thre, offset=THRE_OFFSET, inv=True, visibility=False)

    def pnt_filter(bin_img):
        cnts, _ = cv.findContours(bin_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)
        centers = []
        area_ratio = 0.75
        while len(centers) != 1:
            for cnt in cnts:
                area = cv.contourArea(cnt)
                center, radius = cv.minEnclosingCircle(cnt)
                if area / np.pi / np.power(radius, 2) > area_ratio:
                    centers.append(center)
            if len(centers) > 1:
                area_ratio += 0.5
            elif len(centers) < 1:
                area_ratio -= 0.5

        return centers[0]

    w_pnt = pnt_filter(white)
    b_pnt = pnt_filter(black)

    if visibility:
        tmp = cell.copy()
        tmp = cv.cvtColor(tmp, cv.COLOR_GRAY2BGR)
        cv.circle(tmp, tuple(np.int0(w_pnt)), 2, (0, 0, 255), -1)
        cv.circle(tmp, tuple(np.int0(b_pnt)), 2, (0, 0, 255), -1)
        cv.imshow('cell', tmp)
        cv.waitKey(25)

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
        tmp = cv.drawKeypoints(tmp, tl_pnts_tmp, None, (0, 255, 0))

        w_pnts_tmp = w_pnts.tolist()
        w_pnts_tmp = [cv.KeyPoint(c[0], c[1], 1) for c in w_pnts_tmp]
        tmp = cv.drawKeypoints(tmp, w_pnts_tmp, None, (0, 0, 255))

        b_pnts_tmp = b_pnts.tolist()
        b_pnts_tmp = [cv.KeyPoint(c[0], c[1], 1) for c in b_pnts_tmp]
        tmp = cv.drawKeypoints(tmp, b_pnts_tmp, None, (255, 0, 0))

        tmp = cv.resize(tmp, None, fx=0.4, fy=0.4, interpolation=cv.INTER_AREA)
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

    roi_row_start = decay_pnts_row // 2
    roi_row_end = roi_row_start + res_pnts_row

    roi_col_start = decay_pnts_col // 2
    roi_col_end = roi_col_start + res_pnts_col

    return pnts[roi_row_start:roi_row_end, roi_col_start:roi_col_end]


def get_cvtcoef_pixel_mm(pnts, mm_space):
    """
    The function is to get the coefficients of converting pixel to mm
    :param pnts: recommend using black points to compute cvtcoef
    :param mm_space: the format is (row, col), unit is mm
    :return: return average cvtcoef
    """
    r_mm_space, c_mm_space = mm_space
    roi = get_roi_pnts(pnts, decay_factor=0.4)
    row_pixel_space = np.diff(roi, axis=0)[:, :, 1]
    col_pixel_space = np.diff(roi, axis=1)[:, :, 0]
    cvtcoef = np.mean(np.mean(row_pixel_space) / r_mm_space + np.mean(col_pixel_space) / c_mm_space)

    return cvtcoef


def transform_pnts(b_pnts, w_pnts):
    # moving black points based on white points
    roi_pnts_b = get_roi_pnts(b_pnts)
    roi_pnts_w = get_roi_pnts(w_pnts)

    mean_center_b = np.mean(roi_pnts_b, axis=(0, 1))
    mean_center_w = np.mean(roi_pnts_w, axis=(0, 1))

    center_bias = mean_center_w - mean_center_b
    print(f'Center deviation: dx, dy= ({center_bias[0]:.2f}, {center_bias[1]:.2f})')

    # compute angles
    b_angle = calc_angle(roi_pnts_b)
    w_angle = calc_angle(roi_pnts_w)
    print(f'Angle deviation: b, w= ({b_angle:.3f}, {w_angle:.3f})')

    # rotate white and black points to Cartesian coordinates
    rot_mat_b = cv.getRotationMatrix2D(mean_center_b.tolist(), b_angle, scale=1)
    rot_mat_w = cv.getRotationMatrix2D(mean_center_b.tolist(), w_angle, scale=1)
    # print(rot_mat_b, rot_mat_w)

    # move and rotate all points
    w_pnts -= center_bias

    b_pnts = b_pnts.reshape(-1, 2).T
    b_pnts = np.insert(b_pnts, 2, values=1, axis=0)
    w_pnts = w_pnts.reshape(-1, 2).T
    w_pnts = np.insert(w_pnts, 2, values=1, axis=0)
    b_pnts = np.matmul(rot_mat_b, b_pnts).T
    w_pnts = np.matmul(rot_mat_w, w_pnts).T

    return w_pnts, b_pnts, mean_center_b, mean_center_w, b_angle, w_angle


def show_calib_res(black, white, mean_center_b, mean_center_w, b_angle, w_angle, tr_b_pnts, tr_w_pnts):
    # visualization: show result
    row, col = black.shape[:2]
    rotated_matrix = cv.getRotationMatrix2D(mean_center_b.tolist(), b_angle, scale=1)
    rot_im_b = cv.warpAffine(black, rotated_matrix, (col, row), borderMode=cv.BORDER_CONSTANT, borderValue=255)
    cv.drawMarker(rot_im_b, np.int32(mean_center_b.tolist()), 0, cv.MARKER_TILTED_CROSS, 50, 5)

    tx, ty = -(mean_center_w - mean_center_b)
    rot_mtx = cv.getRotationMatrix2D(mean_center_w.tolist(), w_angle, scale=1)
    tr_mtx = np.matrix([[0, 0, tx],
                        [0, 0, ty]])
    tr_mtx += rot_mtx
    tr_im_w = cv.warpAffine(white, tr_mtx, (col, row), borderMode=cv.BORDER_CONSTANT, borderValue=255)

    tr_b_pnts = np.insert(tr_b_pnts, 2, values=0, axis=1).astype(np.float32)
    tr_w_pnts = np.expand_dims(tr_w_pnts, 1).astype(np.float32)

    # method 1: use homography
    H, _ = cv.findHomography(tr_w_pnts, tr_b_pnts)
    tr_im_w = cv.warpPerspective(tr_im_w, H, (col, row), flags=cv.INTER_CUBIC)

    # method 2: use calibrate camera
    INIT_CAM_MAT = False
    mtx = None
    flag = cv.CALIB_USE_QR  # cv.CALIB_FIX_PRINCIPAL_POINT
    if INIT_CAM_MAT:
        f = 25  # focal length
        cmos = np.array([12.8, 9.6])  # CMOS size
        resolution = np.array([5472, 3648])  # max resolution
        dx, dy = cmos / resolution
        cx, cy = resolution / 2
        mtx = np.array([[f / dx, 0, cx], [0, f / dy, cy], [0, 0, 1]]).astype(np.float32)
        flag = cv.CALIB_USE_INTRINSIC_GUESS + cv.CALIB_FIX_PRINCIPAL_POINT + cv.CALIB_RATIONAL_MODEL
        print(f'init cameraMat:\n{mtx}')

    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 50, 1e-4)
    retval = cv.calibrateCameraRO([tr_b_pnts], [tr_w_pnts], (col, row), 1, mtx, None, flags=flag, criteria=criteria)
    rmse, mtx, dists, rvecs, tvecs, _ = retval
    print(f'RMSE: {rmse}')

    new_mtx, _ = cv.getOptimalNewCameraMatrix(mtx, dists, (col, row), 1, (col, row))
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dists, None, new_mtx, (col, row), cv.CV_32FC1)
    # tr_im_w = cv.remap(tr_im_w, mapx, mapy, cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT, borderValue=0)

    # visualization: show result
    rot_im_b = cv.cvtColor(rot_im_b, cv.COLOR_GRAY2BGR)
    tr_im_w = cv.cvtColor(tr_im_w, cv.COLOR_GRAY2BGR)
    rot_im_b[:, :, 2] = 255
    rot_im_b[:, :, 0] = 255
    tr_im_w[:, :, 0] = 255
    tr_im_w[:, :, 1] = 255

    add = cv.addWeighted(cv.bitwise_not(rot_im_b), 1, cv.bitwise_not(tr_im_w), 1, 0)
    # cv.imwrite('add.png', add)
    temp = cv.resize(add, None, fx=0.31, fy=0.31, interpolation=cv.INTER_AREA)
    cv.imshow('temp', temp)
    cv.waitKey()
    cv.destroyWindow('temp')


def undistort_image(cvt_pnts_dev):
    src = cv.imread('./_data/A2D_calib.png', cv.IMREAD_GRAYSCALE)
    # src = cv.imread('calib_output.png', cv.IMREAD_GRAYSCALE)
    row, col = src.shape[:2]

    origin_pnts = imtemplate.draw_calibboard((1080, 1920), (16, 16))
    undistorted_pnts = origin_pnts - cvt_pnts_dev

    undistorted_pnts = np.insert(undistorted_pnts, 2, values=0, axis=1).astype(np.float32)
    origin_pnts = np.expand_dims(origin_pnts, 1).astype(np.float32)

    # method 1: use homography
    H, _ = cv.findHomography(origin_pnts, undistorted_pnts, cv.FM_LMEDS, ransacReprojThreshold=1)
    calib_src = cv.warpPerspective(src, H, (col, row), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT,
                                   borderValue=0)

    # method 2: use calibrate camera
    flag = cv.CALIB_USE_QR  # cv.CALIB_FIX_PRINCIPAL_POINT
    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 50, 1e-4)
    retval = cv.calibrateCameraRO([undistorted_pnts], [origin_pnts], (col, row), 1, None, None, flags=flag,
                                  criteria=criteria)
    rmse, mtx, dists, rvecs, tvecs, _ = retval
    print(f'RMSE: {rmse}')

    new_mtx, _ = cv.getOptimalNewCameraMatrix(mtx, dists, (col, row), 1, (col, row))
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dists, None, new_mtx, (col, row), cv.CV_32FC1)
    # calib_src = cv.remap(src, mapx, mapy, cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT, borderValue=0)

    cv.imshow('calib', calib_src)
    cv.imwrite('calib_output.png', calib_src)
    cv.waitKey()
    cv.destroyWindow('calib')
    return calib_src


def calibrate():
    calib_shape = (16, 16)
    mm_space = (5.4, 9.6)
    cvt_coef_mm_pixel = 144.3 / 1920
    calib_img = cv.imread('calib.bmp', cv.IMREAD_GRAYSCALE)
    white_img = cv.imread('w.bmp', cv.IMREAD_GRAYSCALE)
    roi = extractROI(white_img, calib_img, calib_shape, visibility=False)

    # Method1: extract points using global threshold
    # w_pnts, b_pnts, white, black = extract_pnts_with_circle(roi, calib_shape, False)

    # Method2: extract points using local threshold
    w_pnts, b_pnts = extract_pnts_with_mesh(roi, calib_shape, True)

    # compute cvtcoef from pixel to mm
    cvtcoef = get_cvtcoef_pixel_mm(b_pnts, mm_space)

    # transform such as move and rotate black and white points in order to alignment
    tr_w_pnts, tr_b_pnts, mean_center_b, mean_center_w, b_angle, w_angle = transform_pnts(b_pnts, w_pnts)
    pnts_deviation = tr_w_pnts - tr_b_pnts
    print(f'Max deviation: {abs(pnts_deviation / cvtcoef).max()}')

    # visualization
    # show_calib_res(black, white, mean_center_b, mean_center_w, b_angle, w_angle, tr_b_pnts, tr_w_pnts)

    # write json
    cvt_pnts_dev = pnts_deviation / cvtcoef
    cvt_pnts_dev = cvt_pnts_dev.tolist()
    title = ['dx', 'dy']
    calib_data = dict()
    calib_data['map'] = [dict(zip(title, p)) for p in cvt_pnts_dev]
    calib_data['col'] = 16
    calib_data['row'] = 16
    calib_data['num'] = 256
    read_write_json.write('calibrate.json', calib_data)

    # undistort origin image to generate undistorted image
    # cvt_pnts_dev = pnts_deviation / cvtcoef / cvt_coef_mm_pixel
    # undistort_image(cvt_pnts_dev)


if __name__ == '__main__':
    calibrate()

    # read xml to remap
    # fs_x = cv.FileStorage('mapx.xml', flags=cv.FILE_STORAGE_FORMAT_XML)
    # mapx = fs_x.getNode('mapx').mat()
    # fs_y = cv.FileStorage('mapy.xml', flags=cv.FILE_STORAGE_FORMAT_XML)
    # mapy = fs_y.getNode('mapy').mat()
    # a2d_calib = cv.imread('./_data/A2D_calib.png', cv.IMREAD_GRAYSCALE)
    # calib_out = cv.remap(a2d_calib, mapx, mapy, cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT, borderValue=0)
    # cv.imwrite('calib_out_2nd.png', calib_out)
