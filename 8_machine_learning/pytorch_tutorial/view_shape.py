# coding: utf-8

import torch
import numpy as np

# [batch, channel, height, width]
a = torch.rand(4, 1, 28, 28)
print(a.shape)

print(a.view(4, 28 * 28).shape)

# squeeze and unsqueeze
print(a.unsqueeze(0).shape)

b = torch.tensor([1.2, 3.4])
print(b.shape, b.unsqueeze(-1), b.unsqueeze(-1).shape)
print(b.shape, b.unsqueeze(0), b.unsqueeze(0).shape)

b = torch.rand(32)
f = torch.rand(4, 32, 14, 14)

# [1, 32, 1, 1]
b = b.unsqueeze(0).unsqueeze(2).unsqueeze(-1)
print(b.shape)

# squeeze, shape with dim = 1 will be squeezed
print(b.squeeze().shape)

# expand / repeat(memory copied)
print(b.expand(4, 32, 4, 4).shape)
print(b.expand(4, -1, 4, 4).shape)
print(b.repeat(4, 1, 4, 4).shape)  # number of repeating

# transpose / permute
b = b.expand(4, 32, 32, 3)
b1 = b.transpose(1, 3).contiguous().view(4, 32*32*3).view(4, 3, 32, 32).transpose(1,3)
print(torch.all(torch.eq(b,b1)))

b2 = b.permute(0, 2, 3, 1)
print(b2.shape)