import torch
import numpy as np
import csv

bikes_numpy = np.loadtxt('hour-fixed.csv',
                         dtype=np.float32,
                         delimiter=',',
                         skiprows=1,
                         converters={1: lambda x: float(x[8:10])}
                         )
bikes = torch.from_numpy(bikes_numpy)
print(bikes)

col_list = next(csv.reader(open('hour-fixed.csv'), delimiter=','))
print(bikes.shape)
print(col_list)

daily_bikes = bikes.view(-1, 24, bikes.shape[1])
print(daily_bikes.shape)

daily_bikes = daily_bikes.transpose(1, 2)
print(daily_bikes.shape)

first_day = bikes[:24].long()
weather_onehot = torch.zeros(first_day.shape[0], 4)
print(first_day[:, 9])

# one-hot
weather_onehot.scatter_(dim=1, index=first_day[:, 9].unsqueeze(1) - 1, value=1.0)
# print(weather_onehot)

tmp = torch.cat((bikes[:24], weather_onehot), 1)[: 1]
print(tmp)

daily_weather_onehot = torch.zeros(daily_bikes.shape[0], 4, daily_bikes.shape[2])
daily_weather_onehot.scatter_(1, daily_bikes[:, 9, :].long().unsqueeze(1) - 1, 1.0)
print(daily_weather_onehot.shape)

daily_bikes[:, 9, :] = (daily_bikes[:, 9, :] - 1.0) / 3.0

"""
重新縮放变量为 [0.0, 1.0]
两种方式：
"""

temp = daily_bikes[:, 10, :]
temp_min = torch.min(temp)
temp_max = torch.max(temp)
daily_bikes[:, 10, :] = (daily_bikes[:, 10, :] - temp_min) / (temp_max - temp_min)


temp = daily_bikes[:, 10, :]
daily_bikes[:, 10, :] = (daily_bikes[:, 10, :] - torch.mean(temp)) / torch.std(temp)


