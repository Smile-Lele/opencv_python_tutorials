# coding: utf-8

# coding: utf-8
import os

import numpy as np
import cv2 as cv
from mypackage.multiplot import multiplot as mplt
from matplotlib import pyplot as plt


COLORS = [(48, 48, 255),
              (0, 165, 255),
              (0, 255, 0),
              (255, 255, 0),
              (147, 20, 255),
              (144, 238, 144)]


def detect_drop(src_img):
    imdict = dict()
    # src_img = cv.imread('../mydata/drop_2.png')
    img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
    img = cv.fastNlMeansDenoising(img, None, 5, 7, 21)
    img = cv.GaussianBlur(img, (5, 5), 0)

    retval, labels, stats, _ = cv.connectedComponentsWithStatsWithAlgorithm(img, connectivity=8,
                                                                            ltype=cv.CV_16U, ccltype=cv.CCL_WU)
    mask = np.zeros(src_img.shape[:2], np.uint8)

    area = [stats[i][-1] for i in range(retval)]
    maxarea_idx = area.index(max(area))

    mask[labels == maxarea_idx] = 255

    or_mask_gray = cv.bitwise_and(img, mask)
    imdict['or_mask_gray'] = or_mask_gray

    # create a CLAHE object (Arguments are optional).
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_im = clahe.apply(or_mask_gray)

    # res = np.vstack((img, clahe_im))
    # imdict['clahe_im'] = clahe_im
    #
    # hist = cv.calcHist([clahe_im], [0], mask=None, histSize=[256], ranges=[0, 256])
    #
    # hist = hist.squeeze()
    # hist_list = hist.tolist()
    # MAX_GRAYSCALE = hist_list.index(max(hist_list))
    # imdict['hist'] = hist

    ret, thresh = cv.threshold(clahe_im, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    imdict['thresh'] = thresh

    # noise removal
    kernel = np.ones((7, 7), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    imdict['opening'] = opening

    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=5)
    imdict['sure_bg'] = sure_bg

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5, dstType=cv.CV_32F)
    imdict['dist_transform'] = dist_transform

    ret, sure_fg = cv.threshold(dist_transform, 0.3*dist_transform.max(), 255, cv.THRESH_BINARY)
    imdict['sure_fg'] = sure_fg



    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    imdict['unknown'] = unknown

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # markers = cv.applyColorMap(markers, cv.COLORMAP_JET)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv.watershed(src_img, markers)
    imdict['jet_markers'] = markers

    src_img[markers == -1] = [0, 255, 0]
    imdict['img'] = src_img
    # mplt.show(imdict)

    return src_img


if __name__ == '__main__':
    # detect_drop()
    video_path = '../mydata'
    video_name = 'drops.avi'
    path = os.path.join(video_path, video_name)
    video = cv.VideoCapture(path)
    fourcc = int(video.get(cv.CAP_PROP_FOURCC))
    fps = video.get(cv.CAP_PROP_FPS)
    frame_size = (np.int0(video.get(cv.CAP_PROP_FRAME_WIDTH)), np.int0(video.get(cv.CAP_PROP_FRAME_HEIGHT)))
    out = cv.VideoWriter(os.path.join(video_path, 'output.avi'), fourcc, fps, frame_size)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            out.release()
            video.release()
            cv.destroyAllWindows()
            exit(0)

        frame = detect_drop(frame)
        out.write(frame)
        # print(frame.shape)
        cv.imshow('', frame)

        if cv.waitKey(3) & 0xFF == ord('q'):
            break

    out.release()
    video.release()
    cv.destroyAllWindows()
