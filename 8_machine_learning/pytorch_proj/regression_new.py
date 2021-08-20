# coding: utf-8

import torch
import torch.nn as nn
from matplotlib import pyplot as plt


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    x = torch.unsqueeze(torch.linspace(-3, 3, 250), dim=1)  # torch requires 2 dim
    y = x.pow(2) + 0.8 * torch.rand(x.size())
    print(x)
    model = Model()

    plt.ion()
    plt.show()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.05)
    loss_func = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

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
