import multiprocessing
import os

import cv2 as cv
import numpy as np


def main():
    img_dicts = dict()

    img_path = '/Users/smile/Desktop/'
    img_name = '2.png'

    src_img = cv.imread(os.path.join(img_path, img_name))
    if src_img is None or len(src_img) == 0:
        print('no image was found')
        exit(0)

    height, width = src_img.shape[:2]
    gray_img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
    img_dicts['gray_img'] = gray_img

    retval, labels, stats, _ = cv.connectedComponentsWithStatsWithAlgorithm(gray_img, connectivity=8,
                                                                            ltype=cv.CV_16U, ccltype=cv.CCL_WU)
    masked_img = np.zeros((height, width, 1), np.uint8)

    def sort_contours(index):
        return stats[index][-1]

    idx_contours = [i for i in range(retval)]
    idx_contours.sort(key=sort_contours, reverse=True)

    max_area_label = idx_contours[0]
    mask = labels == max_area_label
    mask = ~mask
    masked_img[mask] = 255
    masked_img = cv.dilate(masked_img, (17, 17), iterations=10)
    img_dicts['masked_img'] = masked_img

    or_mask_gray = cv.bitwise_or(masked_img, gray_img)
    img_dicts['or_mask_gray'] = or_mask_gray

    # show_imgs(img_dicts)

    hist = cv.calcHist([or_mask_gray], [0], None, [256], [0, 256])
    # img_dicts['hist'] = hist.ravel()

    # bilater_img = cv.bilateralFilter(gray_img, 9, 150, 150, cv.BORDER_CONSTANT)
    # img_dicts['bilater_img'] = bilater_img
    #
    # _, thre = cv.threshold(bilater_img, 50, 255, cv.THRESH_BINARY)
    # img_dicts['thre'] = thre
    #
    # edges = cv.Canny(thre, 50, 255, apertureSize=3)
    # img_dicts['edges'] = edges
    #
    # lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)
    # print(len(lines))
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv.line(src_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    lapla = cv.Laplacian(or_mask_gray, cv.CV_64F, ksize=3)
    lapla = np.uint8(np.absolute(lapla))
    # img_dicts['lapla'] = lapla

    sobelX = cv.Sobel(or_mask_gray, cv.CV_64F, 1, 0)
    sobelY = cv.Sobel(or_mask_gray, cv.CV_64F, 0, 1)

    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))

    # img_dicts['sobelX'] = sobelX
    # img_dicts['sobelY'] = sobelY
    # show_imgs(img_dicts)

    edges = cv.Canny(sobelY, 0, 255, apertureSize=3)
    img_dicts['edges'] = edges
    # show_imgs(img_dicts)


    print(height, width)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=(width//2), maxLineGap=100)

    if lines is None:
        print('no line can be found')
        exit()

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(src_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    img_dicts['src_img'] = src_img
    show_imgs(img_dicts)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
