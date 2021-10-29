import numpy as np
import cv2 as cv
from mypackage.fileUtils import read_write_json as rj
import numpy as np
from matplotlib import pyplot as plt


def remove_anomaly_kmeans(data):
    data = np.asarray(data, dtype=np.float32).reshape(-1, 1)

    # 定义终止标准 = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-3)
    # 设置标志
    flags = cv.KMEANS_RANDOM_CENTERS

    # 应用K均值
    compactness, labels, centers = cv.kmeans(data, 5, None, criteria, 100, flags)
    # print(centers)

    class_index = np.argsort(centers, axis=0)
    med_class_index = class_index[1:4].ravel()

    valid_region = np.zeros_like(labels, bool)
    for i in med_class_index:
        valid_region |= (labels == i)

    A = data[labels == 0]
    B = data[labels == 1]
    C = data[labels == 2]
    D = data[labels == 3]
    E = data[labels == 4]

    plt.scatter(np.argwhere(labels==0)[:, 0], A)
    plt.scatter(np.argwhere(labels==1)[:, 0], B)
    plt.scatter(np.argwhere(labels==2)[:, 0], C)
    plt.scatter(np.argwhere(labels==3)[:, 0], D)
    plt.scatter(np.argwhere(labels==4)[:, 0], E)

    plt.show()

    target_data = data[valid_region]
    retpivot = np.mean(target_data)
    print(target_data)
    return retpivot, target_data


if __name__ == '__main__':
    data = rj.read('temp_list_for_img.json')
    data_size = len(data)
    for d in data[0:data_size:5]:
        retpivot, target_data = remove_anomaly_kmeans(d)
        plt.scatter(list(range(0, len(d))), d)
        plt.hlines(np.mean(d), 0, len(d), 'C0', 'dashed')
        plt.hlines(retpivot, 0, len(d), 'C0')
    plt.show()