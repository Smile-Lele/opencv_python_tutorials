import torch

# PyTorch内部使用pickle来序列化张量对象和实现用于存储的专用序列化代码

points = torch.rand(3, 4)
# print(points)

# save data
torch.save(points, 'out_points.t')

with open('out_points.t', 'wb') as f:
    torch.save(points, f)

# load
points = torch.load('out_points.t')
print(points)

with open('out_points.t', 'rb') as f:
    points = torch.load(f)

print(points)

"""
如果只想通过PyTorch加载张量，则上述例子可让你快速保存张量，
但这个文件格式本身是不互通（interoperable）的，
你无法使用除PyTorch外其他软件读取它
"""
