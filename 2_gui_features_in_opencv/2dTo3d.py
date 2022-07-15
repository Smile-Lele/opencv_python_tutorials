import cv2 as cv
from mypackage.imUtils import icv
from matplotlib import pyplot as plt
import numpy as np


def get_non_zero_coordinate(img, thre=0):
    print(f'{get_non_zero_coordinate.__name__}:{thre=}')
    thre, img = cv.threshold(img, thre, 255, cv.THRESH_BINARY)
    pnts = cv.findNonZero(img)
    val = np.asarray([img[p[1], p[0]] for p in pnts.squeeze()]).reshape(-1, 1)
    pnts = np.asarray(pnts).reshape(-1, 2)
    print(f'{pnts.shape=}')
    return pnts, val


if __name__ == '__main__':
    fig = plt.figure('surface')
    ax = fig.gca(projection='3d')
    img = icv.draw_cross((800, 600), 100, 5)
    imgs = [img for i in range(10)]
    for i, img in enumerate(imgs, 1):
        pnts, val = get_non_zero_coordinate(img, 80)
        val.fill(i)
        ax.scatter(pnts[:, 1], pnts[:, 0], val, c='#1f77b4', marker='.', depthshade=False, linewidths=1, edgecolors='none')
    xticks = np.arange(0, 800, 0.5)
    yticks = np.arange(0, 600, 0.5)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    plt.show()
