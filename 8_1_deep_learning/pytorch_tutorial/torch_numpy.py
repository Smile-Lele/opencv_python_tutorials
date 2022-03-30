import torch
import numpy as np

# Torch与Numpy数组的这种零拷贝互通性，是由于Pytorch遵守Python缓冲协议
points = torch.ones(3, 4)
points_np = points.numpy()
print(points_np)
print(points_np.dtype)

# from_numpy使用相同的缓冲共享策略
points = torch.from_numpy(points_np)
print(points)