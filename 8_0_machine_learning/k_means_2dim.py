import numpy as np
import cv2 as cv
from mypackage.fileUtils import rw_json as rj
import numpy as np
from matplotlib import pyplot as plt


def remove_anomaly_kmeans(expect):
    expect = np.asarray(expect, dtype=np.float32).reshape(1, -1)
    sigma = np.random.randint(-5, 5, expect.shape).astype(np.float32)

    Z = np.vstack((sigma, expect)).T
    print(data.shape)

    # 定义终止标准 = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-3)
    # 设置标志
    flags = cv.KMEANS_RANDOM_CENTERS

    # 应用K均值
    compactness, label, center = cv.kmeans(Z, 3, None, criteria, 100, flags)
    # print(centers)

    A = Z[label.ravel() == 0]
    B = Z[label.ravel() == 1]
    C = Z[label.ravel() == 2]
    # D = Z[label.ravel() == 3]
    # E = Z[label.ravel() == 4]
    # 绘制数据
    plt.scatter(A[:, 0], A[:, 1])
    plt.scatter(B[:, 0], B[:, 1], c='r')
    plt.scatter(C[:, 0], C[:, 1], c='g')
    # plt.scatter(D[:, 0], D[:, 1], c='b')
    # plt.scatter(E[:, 0], E[:, 1])
    plt.scatter(center[:, 0], center[:, 1], s=80, c='y', marker='s')
    plt.xlabel('Height'), plt.ylabel('Weight')
    plt.show()




if __name__ == '__main__':
    data = rj.read('temp_list_for_img.json')
    data = np.asarray(data)
    sigma = np.random.randint(-5, 5, data.shape)
    for e in data:
        remove_anomaly_kmeans(e)