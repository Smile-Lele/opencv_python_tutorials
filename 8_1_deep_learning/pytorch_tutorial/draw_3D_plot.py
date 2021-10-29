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
    print(Z.shape)
    fig = plt.figure('himmelblau')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='turbo')
    ax.view_init(60, -30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


if __name__ == '__main__':
    draw()
    x = torch.rand(1, 2)
    print(x)
    epsilon = 6
    x = torch.multiply(x, 2* epsilon) - epsilon
    x.requires_grad_()
    # torch.nn.init.kaiming_uniform_(x)
    print(x)

    LR = 1e-3
    optimizer = torch.optim.Adam([x], lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True)
    for step in range(50000):

        pred = himmelblau(x.squeeze())

        optimizer.zero_grad()
        pred.backward()
        optimizer.step()

        if step % 2000 == 0:
            print('step {}:x={},f(x)={}'.format(step, [x], pred.item()))
            scheduler.step(pred.item())
            if pred.item() == 0:
                break



