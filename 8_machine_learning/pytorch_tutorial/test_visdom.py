# coding: utf-8

import os
import sys
import cv2 as cv
import numpy as np
import torch
from  visdom import Visdom
from mypackage.multiplot import multiplot as mplt


if __name__ == '__main__':
    # example1
    viz = Visdom()
    viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
    viz.line([loss.item()], [global_step], win='train_loss', update='append')

    # example2
    viz = Visdom()
    # [y1, y2], [x]
    viz.line([0., 0.], [0.], win='test', opts=dict(title='test loss&acc.'), legend=['loss', 'acc.'])
    viz.line([[test_loss, correct/len(test_loader.dataset)]], [global_step], win='test', update='append')


    # example3
    viz = Visdom()
    # data is tensor
    viz.images(data.view(-1, 1, 28, 28), win='x')
    viz.text(str(pred.detach().cpu().numpy()), win='pred', opts=dict(title='pred'))
