# coding: utf-8
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    x = torch.unsqueeze(torch.linspace(-20, 20, 100), dim=1)  # torch requires 2 dim
    print(x.shape, x.size())
    y = x.pow(3) + 0.8 * torch.rand(x.size())

    # x = torch.squeeze(torch.normal(2, 3, size=(1, 2000)), dim=1).T
    # x, _ = torch.sort(x, 0)
    # sigma, mean = torch.std_mean(x)
    # y = torch.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))) / (torch.sqrt(2 * torch.tensor(np.pi)) * sigma)

    model = Model()

    plt.ion()
    plt.show()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    loss_func = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for t in range(10000):
        pred = model(x)

        loss = loss_func(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 5 == 0:
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), pred.data.numpy(), 'r-', lw=2)
            plt.pause(0.1)
            print(f'Loss={loss.data:.4f}')
            scheduler.step(loss.data)

    plt.ioff()
    plt.show()
