import time

import cv2 as cv
import numpy as np


def extract_items(grayImg: np.uint8, minarea: int):
    """
    This is a quick method to extract each component in a grayscale image
    :param grayImg: grayscale image
    :param minarea: minimum area of circle
    """
    print(f'grayImg:\n\tshape:{grayImg.shape},ndim:{grayImg.ndim},dtype:{grayImg.dtype}')
    retval, labels, stats, centroids = cv.connectedComponentsWithStatsWithAlgorithm(grayImg, 8, cv.CV_32S, cv.CCL_BBDT)
    print(f'items except background:{retval - 1}')
    sorted_index = sorted(list(range(1, retval)), key=lambda x: stats[x, cv.CC_STAT_AREA], reverse=True)
    canvas = np.zeros(grayImg.shape, dtype=grayImg.dtype)

    for i in sorted_index:
        area = stats[i, cv.CC_STAT_AREA]
        if area < minarea:
            continue

        canvas[labels == i] = 255

        # TODO: do something
        cv.imshow('', canvas)
        cv.waitKey(3000)

        canvas.fill(0)


if __name__ == '__main__':
    img = cv.imread('S000163_P1.png', cv.IMREAD_GRAYSCALE)
    start_t = time.time()
    extract_items(img, 0)
    print(time.time() - start_t)
