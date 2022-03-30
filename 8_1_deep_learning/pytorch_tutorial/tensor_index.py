import torch

points = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(points[1:])
print(points[1:, :])
print(points[1:, 0])
