import itertools
import multiprocessing
import os
import random
from concurrent import futures

import cv2 as cv
import numpy as np
import pandas as pd
import tqdm

from mypackage.fileUtils import zipper
from mypackage.imUtils import icv 
from mypackage.strUtils import str_utils
from mypackage.timeUtils import timer

COLORS = np.random.randint(64, 255, size=(100, 3)).tolist()


def extract_items_dict(file, minArea=0, reverseOrder=True) -> dict:
    img = icv.imread_ex(file, cv.IMREAD_GRAYSCALE)
    gray = icv.cvtColor2Gray(img)
    ret = cv.connectedComponentsWithStatsWithAlgorithm(gray, 8, cv.CV_32S, cv.CCL_BBDT)
    items_total_num, labels, stats, centroids = ret

    # remove tiny items
    cnts_idx = list(range(1, items_total_num))
    valid_cnts_idx = list(filter(lambda l: stats[l, cv.CC_STAT_AREA] > minArea, cnts_idx))

    # sort items based on area
    items_valid_num = len(valid_cnts_idx)
    valid_cnts_idx.sort(key=lambda id: stats[id, cv.CC_STAT_AREA], reverse=reverseOrder)

    canvas = np.zeros_like(gray)
    items_dict_ = dict()
    for i, label in enumerate(valid_cnts_idx, 1):
        item_info = {}
        # extract item
        canvas[labels == label] = 255
        item = canvas.copy()
        item_info['img'] = item
        canvas.fill(0)

        # extract area
        area = stats[label, cv.CC_STAT_AREA]
        item_info['itemArea'] = area

        # extract bounding box
        x = stats[label, cv.CC_STAT_LEFT]
        y = stats[label, cv.CC_STAT_TOP]
        w = stats[label, cv.CC_STAT_WIDTH]
        h = stats[label, cv.CC_STAT_HEIGHT]
        boundingbox = (x, y, w, h)
        item_info['boundingBox'] = boundingbox

        # extract centroids
        c_x = centroids[label, 0]
        c_y = centroids[label, 1]
        center = c_x, c_y
        item_info['center'] = center

        # extract contours
        cnts, hier = cv.findContours(item, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)
        item_info['contours'] = cnts[0]

        # minRect
        minRect = cv.minAreaRect(cnts[0])
        # minRect_cx, minRect_cy = minRect[0]
        # minRect_w, minRect_h = minRect[1]
        # _theta = minRect[2]
        # angle = [_theta, _theta - 90][minRect_w < minRect_h]
        # boxCors = np.int0(cv.boxPoints(minRect))
        item_info['rectArea'] = minRect

        # arcLength
        perimeter = cv.arcLength(cnts[0], True)
        item_info['perimeter'] = perimeter

        # save data
        items_dict_[str(i)] = item_info
        del item_info

    # save immediate data for debug
    centers = {str(n): items_dict_[str(n)]['center'] for n in range(1, items_valid_num + 1)}
    drawing(file, centers)

    # print(f'{items_valid_num=}')
    return items_valid_num, items_dict_


def drawing(file, centers):
    img = icv.imread_ex(file, cv.IMREAD_COLOR)
    [cv.putText(img, 'M' + key,
                (val[0].astype(np.int0), val[1].astype(np.int0)),
                cv.FONT_HERSHEY_TRIPLEX, 1, random.choice(COLORS), 1) for key, val in centers.items()]

    dir_, fname_ext, fname = str_utils.split_dir(file)
    data_dir = os.path.join(dir_, 'data')
    str_utils.check_make_dir(data_dir)

    cv.imwrite(os.path.join(data_dir, 'copy_' + fname_ext), img)


def select_items(items: dict, item_sn: list):
    if len(item_sn) > len(items):
        print('input sn out of bound')

    dst = 0
    cnts = []
    for n in item_sn:
        item = items[str(n)]['img']
        cnts.append(items[str(n)]['contours'])
        dst += item

    # visualize
    # dst_viz = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    # cv.drawContours(dst_viz, cnts, -1, random.choice(COLORS), 1)
    # dst_viz = icv.resize_for_display(dst_viz)
    # cv.imshow('', dst_viz)
    # cv.waitKey(30)

    return dst


def morph(src):
    gray = icv.cvtColor2Gray(src)
    gray = cv.resize(gray, None, fx=0.3, fy=0.3, interpolation=cv.INTER_AREA)

    ksize = 1
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1))

    MAX_ITER = 50
    dilate_counter = 0
    for i in range(MAX_ITER):
        # dilate img
        gray = cv.morphologyEx(gray, cv.MORPH_DILATE, kernel)
        dilate_counter += 1

        # check the number of items
        ret = cv.connectedComponentsWithStatsWithAlgorithm(gray, 8, cv.CV_32S, cv.CCL_BBDT)
        items_num, labels, stats, centroids = ret
        if items_num == 2:
            break
    return dilate_counter


def calcDist_items_pair(items_dict, sn):
    res = {}
    img = select_items(items_dict, sn)
    dilate_counter = morph(img)
    key = 'M' + str(sn[0]) + '_' + str(sn[1])
    res[key] = dilate_counter
    return res


def calcDist_items_pairs_with_process(items_dict, comb_sn):
    future_list = list()
    with futures.ProcessPoolExecutor() as executor:
        for c in comb_sn:
            future = executor.submit(calcDist_items_pair, items_dict, c)
            future_list.append(future)
        done_iter = futures.as_completed(future_list)

    return [it.result() for it in done_iter]


def calcDist_items_pairs(items_dict, comb_sn):
    res_ = {}
    [res_.update(calcDist_items_pair(items_dict, c)) for c in comb_sn]
    return res_


def process_single_image(file):
    items_num, items_dict = extract_items_dict(file, minArea=50)

    TOPn = 5
    TOPn = TOPn if TOPn < items_num else items_num + 1
    item_sn = list(range(1, TOPn))
    comb_sn = [i for i in itertools.combinations(item_sn, 2)]
    dists = calcDist_items_pairs(items_dict, comb_sn)

    # insert image name
    dists.update({'Image': str_utils.split_dir(file)[-1]})

    return dists


def process_images(dir_):
    files = str_utils.read_multifiles(dir_, 'png')

    filtered_files = list(filter(lambda f: os.path.split(f)[-1].endswith('_P1.png'), files))
    return [process_single_image(file) for file in tqdm.tqdm(filtered_files, desc='Process')]


@timer.clock
def process_images_with_process(dir_):
    files = str_utils.read_multifiles(dir_, 'png')

    filtered_files = list(filter(lambda f: os.path.split(f)[-1].endswith('_P1.png'), files))

    future_list = []
    with futures.ProcessPoolExecutor() as executor:
        for file in filtered_files:
            future = executor.submit(process_single_image, file)
            future_list.append(future)
        done_iter = futures.as_completed(future_list)
        done_iter = tqdm.tqdm(done_iter, total=len(filtered_files), desc='Process')

    return [it.result() for it in done_iter]


def save_data(file, data):
    assert len(data) != 0, 'data is Empty'
    df = pd.DataFrame(data)
    df.fillna(value='Nan', inplace=True)
    img_col = df.pop('Image')
    df.insert(0, 'Image', img_col)

    data_dir = os.path.join(file, 'data')
    str_utils.check_make_dir(data_dir)

    df.to_csv(os.path.join(data_dir, 'dist_data.csv'), encoding='utf-8', index=False)


def process_single_model(file):
    dir_, fname_ext, fname = str_utils.split_dir(file)
    data_dir = os.path.join(dir_, fname)
    str_utils.check_make_dir(data_dir)

    # unzip
    zipper.decompress_ultm(file, data_dir)

    # process
    res = process_images_with_process(data_dir)

    # save res
    save_data(data_dir, res)


@timer.clock
def process_models(path):
    files = str_utils.read_multifiles(path, 'ult')
    [process_single_model(file) for file in files]


if __name__ == '__main__':
    multiprocessing.freeze_support()

    path = 'D:/MyData/Model/Model'
    # path = os.getcwd()
    process_models(path)

    # for test one image
    # process_single_model(os.getcwd())
