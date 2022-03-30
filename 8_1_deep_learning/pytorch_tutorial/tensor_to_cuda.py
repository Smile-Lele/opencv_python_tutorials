import torch

print(torch.cuda.is_available())
points_gpu = torch.tensor([[1, 2], [3, 5], [3, 6]], device='cuda:0')
print(points_gpu)

points = torch.tensor([[1, 4], [2, 4], [4, 6]])
points_gpu1 = 4 * points.to(device='cuda:0')
print(points_gpu1)

"""
GPU上运算的结果并不会返回到CPU，如果要将Tensor返回到CPU，
需要用到 to(device='cpu')
"""
points_cpu = points_gpu1.to(device='cpu')
print(points_cpu)