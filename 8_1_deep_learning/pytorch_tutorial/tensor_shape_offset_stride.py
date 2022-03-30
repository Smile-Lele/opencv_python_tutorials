import torch

points = torch.tensor([[1.0, 3], [2, 3], [4, 7]])
second_point = points[1]
print(second_point.storage_offset())
print(second_point.size())
print(second_point.shape)
print(points.stride())

# clone
second_point = points[1].clone()
second_point[0] = 10.0
print(points)

# transpose
points_t = points.t()
print(points_t)

# storage
print(id(points.storage()) == id(points_t.storage()))

# stride
print(points.stride(), points_t.stride())

# Transpose
some_tensor = torch.ones(3,4,5)
print(some_tensor.shape)
print(some_tensor.stride())

some_tensor_t = some_tensor.transpose(0, 2)
print(some_tensor_t.shape)
print(some_tensor_t.stride())

# Contiguous, 新存储对元素进行了重组，因此步长stride也发生了变化，以反映新的布局
print(points.is_contiguous())
print(points_t.is_contiguous())

points = torch.tensor([[1, 4], [2, 5], [3, 6]])
points_t = points.t()
print(points_t)
print(points_t.storage())
print(points_t.stride())

points_t_conti = points_t.contiguous()
print(points_t_conti)
print(points_t_conti.stride())
print(points_t_conti.storage())

