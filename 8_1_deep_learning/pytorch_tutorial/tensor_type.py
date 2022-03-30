import torch

print(f'{torch.cuda.is_available()=}')

# dtype 1
double_points = torch.ones(10, 2, dtype=torch.double)
shrot_points = torch.tensor([[1, 3],[3, 5]], dtype=torch.short)

print(shrot_points.dtype)
print(double_points.dtype)

# dtype 2
double_points = torch.zeros(10, 2).double()
short_points = torch.tensor([[1, 2], [2, 3]]).short()
print(short_points.dtype)
print(double_points.dtype)

#  dtype 3
double_points = torch.zeros(10, 2).to(torch.double)
short_points = torch.tensor([[1, 2], [3, 4]]).to(dtype=torch.short)
print(short_points.dtype)
print(double_points.dtype)

# dtype 4
points = torch.randn(10, 2)
short_points = points.type(torch.short)
print(short_points.dtype)

