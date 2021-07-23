import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
from matplotlib import pyplot as plt
import multiprocessing

# hyper parameters
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12


def run():
    x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
    y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))

    # plt.scatter(x.data.numpy(), y.data.numpy())
    # plt.show()

    torch_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2)

    net_SGD = torch.nn.Sequential(
        torch.nn.Linear(1, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 1),
    )

    net_Momentum = torch.nn.Sequential(
        torch.nn.Linear(1, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 1),
    )

    net_RMSprop = torch.nn.Sequential(
        torch.nn.Linear(1, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 1),
    )

    net_Adam = torch.nn.Sequential(
        torch.nn.Linear(1, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 1),
    )

    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

    opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
    opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

    loss_func = torch.nn.MSELoss()
    losses_his = [[], [], [], []]

    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader):
            b_x = Variable(batch_x)
            b_y = Variable(batch_y)

            for net, opt, l_his in zip(nets, optimizers, losses_his):
                output = net(b_x)
                loss = loss_func(output, b_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                l_his.append(loss.data)

    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    for label, l_his in zip(labels, losses_his):
        plt.plot(l_his, label=label)
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim(0, 0.2)
    plt.show()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    run()
