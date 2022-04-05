# coding: utf-8

import torch

a = torch.rand(3, 4)
b = torch.rand(4)

print(a @ b)

print(a.clamp(0.8, 0.95))
