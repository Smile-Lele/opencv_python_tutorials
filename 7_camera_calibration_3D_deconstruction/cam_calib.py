# coding:utf-8
import os

import cv2
import numpy as np
import glob
from mypackage.imUtils import ext_json


def undistorting(src, mtx=None, dist=None):
    # 获得新内参
    rows, cols, ch = src.shape
    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (cols, rows), 1, (cols, rows))  # 自由比例参数 between(0, 1)
    # print("新内参：\n", newCameraMtx)

    # 去畸变
    dst = cv2.undistort(src, mtx, dist, None, newCameraMtx)
    return dst


def evaluating(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    # 标定结果评价
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    print("平均误差: ", total_error / len(objpoints))


def calibrating(w, h, images):
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # 找棋盘角点标志位
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

    # 储存棋盘格角点的世界坐标和图像坐标对
    objpoints = []  # 在世界坐标系中的三维点
    imgpoints = []  # 在图像平面的二维点

    if images is None:
        raise FileExistsError('images not existed, please check file path')

    for image in images:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        retval, corners = cv2.findChessboardCornersSB(gray, (w, h), None, cv2.CALIB_CB_ACCURACY)
        if retval:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, (w, h), corners, retval)
            img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            cv2.imshow('findCorners', img)
            cv2.waitKey(1)
    cv2.destroyAllWindows()

    # 标定求解参数
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, None, None, flags=0,
                                                       criteria=criteria)

    para_dict = dict()
    para_dict['mtx'] = mtx.tolist()
    print("内参:\n", mtx)

    # distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    para_dict['dist'] = dist.tolist()
    print("畸变参数:\n", dist)

    ext_json.write('./_data/cam_para.json', para_dict)

    return ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints


def run():
    # w, h = the num of grids -1
    w, h = (12, 10)

    file_path = 'D:/MyData/new_calibration_images/'
    images = glob.glob(file_path + '*.bmp')

    # 1-calibrate camera
    ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints = calibrating(w, h, images)

    # 2-undistort image
    filename = os.path.join('D:/MyData', '1.bmp')
    img = cv2.imread(filename)
    undistorted_im = undistorting(img, mtx, dist)
    cv2.imwrite('D:/MyData/' + '1_r.bmp', undistorted_im)

    # 3-evaluate error
    evaluating(objpoints, imgpoints, rvecs, tvecs, mtx, dist)


if __name__ == '__main__':
    run()
