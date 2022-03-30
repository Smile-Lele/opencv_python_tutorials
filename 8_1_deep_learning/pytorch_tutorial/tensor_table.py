import csv

import numpy as np
import torch

with open('winequality-white.csv', 'rb') as f:
    wineq_numpy = np.loadtxt(f, dtype=np.float32, delimiter=';', skiprows=1)

print(wineq_numpy)
col_list = next(csv.reader(open('winequality-white.csv'), delimiter=';'))
print(wineq_numpy.shape)
print(col_list)

wineq = torch.from_numpy(wineq_numpy)
print(wineq.shape)
print(wineq.type())

"""
3类数值
1. 连续值 continuous, 严格可排序，数值间计算有严格意义
2. 序数 ordinal， 严格可排序，值间相对关系无意义
3. 类别 categorical，没有顺序，没有数值含义, one-hot
"""

# 将ground truth从数据集中分割开
data = wineq[:, :-1]
print(data.shape)
target = wineq[:, -1].long()
print(target)

# one-hot
target_onehot = torch.zeros(target.shape[0], 10)
target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
print(target_onehot)

# normalization
data_mean = torch.mean(data, dim=0)
print(data_mean)

data_var = torch.var(data, dim=0)
print(data_var)

epsilon = 1e-6
data_normalized = (data - data_mean) / (torch.sqrt(data_var) + epsilon)
print(data_normalized)

# target
bad_indexes = torch.le(target, 3)
print(bad_indexes.shape, bad_indexes.dtype, bad_indexes.sum())
bad_data = data[bad_indexes]
print(bad_data)

# data
bad_data = data[torch.le(target, 3)]
mid_data = data[torch.gt(target, 3) & torch.lt(target, 7)]
good_data = data[torch.ge(target, 7)]

bad_mean = torch.mean(bad_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim=0)

for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))

total_sulfur_threshold = 141.83
total_sulfur_data = data[:, 6]
predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)
print(predicted_indexes.shape, predicted_indexes.dtype, predicted_indexes.sum())

# target
actual_indexes = torch.gt(target, 5)
print(actual_indexes.shape, actual_indexes.dtype, actual_indexes.sum())

# evaluation
n_mathches = torch.sum(actual_indexes & predicted_indexes).item()
n_predicted = torch.sum(predicted_indexes).item()
n_actual = torch.sum(actual_indexes).item()
print(n_mathches, n_mathches/n_predicted, n_mathches/n_actual)