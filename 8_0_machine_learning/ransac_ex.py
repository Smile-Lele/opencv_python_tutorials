# coding: utf-8

from mypackage.fileUtils import read_write_json as rj
import numpy as np
from matplotlib import pyplot as plt


def remove_anomaly_ransac(data, sigma):
    d_min, d_max = np.min(data), np.max(data)
    pivot = np.linspace(d_min, d_max+1, num=np.int0(np.ptp(data)))
    max_pnts_num = -1
    retpivot = 0
    data_std = 0
    pre_std = 0
    for p in pivot:
        region_pnts_num = len(data) - (np.sum(data > p + sigma) + np.sum(data < p - sigma))
        print(region_pnts_num)
        min_index = np.argwhere(np.asarray(data) >= p - sigma).ravel()
        max_index = np.argwhere(np.asarray(data) <= p + sigma).ravel()
        union_index = set(min_index) & set(max_index)
        target_data = [data[i] for i in union_index]
        data_std = np.std(target_data)

        if region_pnts_num > max_pnts_num:
            max_pnts_num = region_pnts_num
            retpivot = p
        elif region_pnts_num == max_pnts_num:
            if data_std < pre_std:
                retpivot = p

    min_index = np.argwhere(np.asarray(data) >= retpivot - sigma).ravel()
    max_index = np.argwhere(np.asarray(data) <= retpivot + sigma).ravel()
    union_index = set(min_index) & set(max_index)
    target_data = [data[i] for i in union_index]
    # print(target_data)
    # print(retpivot)
    return target_data, retpivot


if __name__ == '__main__':
    data = rj.read('temp_list_for_img.json')
    sigma = 3
    data_size = len(data)
    for d in data[1:data_size-20:6]:
        target_data, retpivot = remove_anomaly_ransac(d, sigma)
        plt.scatter(list(range(0, len(d))), d)
        plt.hlines(np.mean(target_data), 0, len(d), 'r')
        plt.hlines(retpivot-sigma, 0, len(d), 'C0', 'dashed')
        plt.hlines(retpivot, 0, len(d))
        plt.hlines(retpivot+sigma, 0, len(d), 'C0', 'dashed')
    plt.show()
