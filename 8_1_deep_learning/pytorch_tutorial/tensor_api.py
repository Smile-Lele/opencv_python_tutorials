import torch

"""
torch模块中中函数
"""
a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)

"""
tensor的方法
"""
a = torch.ones(3, 2)
a_t = a.transpose(0, 1)
print(a)
print(a_t)
"""
以上两种操作没有区别，可以互换使用
注意：
有少量的操作仅作为张量对象的方法存在
"""
a = torch.ones(3, 2)
a.zero_()
print(a)
"""
下划线标识，表明该方法是inplace运行的，
就是直接修改输入而不是创建新的输出并返回，
任何不带下滑线的方法都将保持源张量不变并返回新的张量
inplace 叫 就地操作，会修改到源张量内容
"""

# exercise
a = list(range(9))
print(a)
a_tensor = torch.tensor(a)
print(a_tensor)
print(a_tensor.shape)
print(a_tensor.size())
print(a_tensor.stride())
print(a_tensor.storage_offset())
# print(a_tensor.storage())

b = a_tensor.view(3, 3)
print(b[1, 1])

c = b[1:, 1:]
print(c.size())
print(c.storage_offset())
print(c.stride())
print(c)

a_gpu = a_tensor.to('cuda')
cos_a = torch.cos(a_gpu)
sqrt_a = torch.sqrt(a_gpu)
print(cos_a)
print(sqrt_a)
print(a_gpu.type(torch.double).sqrt_())
print(a_gpu.type(torch.double).cos_())
