# coding: utf-8

import torch


a = torch.rand(4, 32, 14, 14)
b = torch.tensor(3)
print(a+b)