import torch
import numpy as np

if __name__ == '__main__':
    print(f'cuda is available:{torch.cuda.is_available()}')
    a = torch.randn(2, 3)
    print(a, type(a), a.type())
    print(isinstance(a, torch.FloatTensor))

    print(isinstance(a.cuda(), torch.cuda.FloatTensor))

    b = np.random.randn(2, 3)
    print(b, type(b))

    t = torch.tensor([[1.0, 2.9, 5], [2, 6, 7]])
    print(t, t.type())
    print(t.shape, t.size(1), t.dim())

    # (batch, channel, height, width)
    a = torch.randn(16, 3, 28, 28)
    print('size:{}'.format(a.shape))
    print('dim:{}'.format(a.dim()))
    print('num:{}'.format(a.numel()))

    a = torch.ones(3)
    print(a)
    print(a.type())

    points = torch.zeros(3, 2)
    print(points[0])
    print(float(points[0, 0]))

    # PyTorch的张量Tensor就是这种存储Storage的视图View, (eg: im2col)
    # Storage存储始终是一维，与引用它的任何张量的维数无关
    points = torch.tensor([[1, 4.0], [2, 1], [2, 5]])
    print(points.type())
    print(points.storage())
