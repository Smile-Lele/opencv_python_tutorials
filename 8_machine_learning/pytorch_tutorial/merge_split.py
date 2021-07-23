# coding: utf-8

import torch


a = torch.rand(32, 8)
b = torch.rand(32, 8)

# merge
print(torch.cat([a, b], dim=1).shape)
c = torch.stack([a, b, a], dim=0)
print(c.shape)


# split
# based on length
print(c.split(2, dim=0)[0].shape)

# based on num of block
print(c.chunk(3, dim=0)[0].shape)
