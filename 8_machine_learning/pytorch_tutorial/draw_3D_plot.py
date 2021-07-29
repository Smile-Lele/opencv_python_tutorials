# coding: utf-8

import os
import sys
import cv2 as cv
import numpy as np
import torch
from mypackage.multiplot import multiplot as mplt
from matplotlib import pyplot as plt


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def draw():
    x = np.arange(-6, 6, 0.1)
    y = np.arange(-6, 6, 0.1)

    print(f'x,y range:{x.shape},{y.shape}')
    X, Y = np.meshgrid(x, y)
    print(f'X,Y maps:{X.shape},{Y.shape}')
    Z = himmelblau([X, Y])
    fig = plt.figure('himmelblau')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='turbo')
    ax.view_init(60, -30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


if __name__ == '__main__':
    # draw()
    x = torch.randn(1, 2, requires_grad=True)
    # x.requires_grad_()
    torch.nn.init.kaiming_normal_(x)
    print(x)

    optimizer = torch.optim.Adam([x], lr=1e-3)
    for step in range(20000):

        pred = himmelblau(x.squeeze())

        optimizer.zero_grad()
        pred.backward()
        optimizer.step()

        if step % 2000 == 0:
            print('step {}:x={},f(x)={}'.format(step, [x], pred.item()))


