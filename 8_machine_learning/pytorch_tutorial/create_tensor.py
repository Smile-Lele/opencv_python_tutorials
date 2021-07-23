import numpy as np
import torch

if __name__ == '__main__':
    a = np.array([2, 3.4])
    a_tensor = torch.from_numpy(a)
    print(a, a_tensor, sep=' | ')

    ones_ = np.ones([2, 3])
    ones_tensor = torch.from_numpy(ones_)
    print(ones_, ones_tensor, sep='\n\n')

    # tensor(numpy() or list())
    b = torch.tensor([[2, 3.2], [1, 2]])
    print(b)

    # Tensor() = FloatTensor(numpy() or list() or shape)
    T = torch.FloatTensor(ones_)
    T_1 = torch.FloatTensor(2, 3)
    print(T, T_1, sep='\n\n')

    # set default tensor type
    torch.set_default_tensor_type(torch.DoubleTensor)
    tensor_ = torch.tensor([1, 2.3])
    print(tensor_.type())

    # Recommend:
    # tensor(numpy() or list())
    # Tensor(shape), note that default tensor type
    full_ = torch.full([2], 8)
    print(full_)

    linear = torch.linspace(0, 10, steps=3)
    log = torch.logspace(0, -1, steps=4)
    print(linear, log, sep=' | ')

    src = torch.tensor([[1, 2, 3], [3, 4, 5]])
    src_take = torch.take(src, torch.tensor([0, 2, 4]))
    print(src_take)
