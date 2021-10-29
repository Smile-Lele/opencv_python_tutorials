# coding: utf-8

import os
import sys
import cv2 as cv
import numpy as np
import torch
from mypackage.multiplot import multiplot as mplt

a = torch.rand(3, 4)
b = torch.rand(4)

print(a @ b)

print(a.clamp(0.8, 0.95))
