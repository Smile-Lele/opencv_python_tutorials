import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from matplotlib import pyplot as plt

# torch.manual_seed(1)  # reproducible

# 假数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)


def save():
    # 建网络
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    # training
    for t in range(100):
        prediction = net(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    torch.save(net, 'net.pkl')  # entire net
    torch.save(net.state_dict(), 'net_param.pkl')  # parameters


def restore_net():
    net = torch.load('net.pkl')
    prediction = net(x)

    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


def restore_param():
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    net.load_state_dict(torch.load('net_param.pkl'))
    prediction = net(x)

    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()

save()

restore_net()

restore_param()