# coding: utf-8

import os
import sys
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from mypackage.imUtils import icv



# 1. gradient
x = torch.ones(1)
w = torch.full([1], 2.0, requires_grad=True)
print(w)
mse = F.mse_loss(torch.ones(1), x*w)
print(mse)
grad_ = torch.autograd.grad(mse, [w])
print(grad_)

# 2. backward
mse = F.mse_loss(torch.ones(1), x*w)
mse.backward()
print(w.grad)

# norm
print(w.norm(2))
print(w.grad.norm(2))