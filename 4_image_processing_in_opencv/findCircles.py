import cv2 as cv
import numpy as np
from mypackage.imUtils import icv

if __name__ == '__main__':
    # src = icv.imread_ex('circle.png', cv.IMREAD_GRAYSCALE)
    # thre, src = cv.threshold(src, 128, 255, cv.THRESH_BINARY_INV)

    src = icv.draw_circleGrid((1080, 1920), (10, 20))

    params = cv.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 200
    params.maxArea = 500
    # params.minDistBetweenBlobs = 200
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.1
    params.filterByCircularity = True
    params.minCircularity = 0.1
    params.filterByConvexity = True
    params.minConvexity = 0.2
    detector = cv.SimpleBlobDetector_create(params)

    # Detect and visualize blobs
    keypnts = detector.detect(src)
    print(f'{len(keypnts)=}')

    src = icv.cvtGray2BGR(src)
    keypnts_img = cv.drawKeypoints(src, keypnts, np.array([]), (0, 0, 255),
                                         cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    keypnts_img = icv.resize_ex(keypnts_img, (540, 960))
    cv.imshow('keypoints', keypnts_img)
    cv.waitKey()
    cv.destroyWindow('keypoints')

    calib_shape = (10, 20)
    # method2 to find points
    flag = cv.CALIB_CB_SYMMETRIC_GRID
    # The function requires white space (like a square-thick border, the wider the better)
    # around the board to make the detection more robust in various environments.
    ret, pnts = cv.findCirclesGrid(src, (calib_shape[1], calib_shape[0]), flags=flag, blobDetector=detector)
    print(f'find points:{len(pnts)}')

    # draw chessboard corners
    temp = src.copy()
    cv.drawChessboardCorners(temp, calib_shape, pnts, True)
    temp = icv.resize_ex(temp, (540, 960))
    cv.imshow('corners', temp)
    cv.waitKey()
    cv.destroyWindow('corners')
