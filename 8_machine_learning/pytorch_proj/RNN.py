# coding: utf-8

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,  # (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        rnn_out, (hidden_n, hidden_c) = self.rnn(x, None)  # (batch, time_step, input_size)
        out = self.out(rnn_out[:, -1, :])
        return out


def train_model(model, device, criteria, train_loader, optimizer, epoch):
    model.train()
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x, b_y = b_x.to(device), b_y.to(device)

        output = model(b_x.view(-1, 28, 28))
        loss = criteria(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f'Epoch:{epoch}, Loss:{loss.item() :.6f}')


def test_model(model, device, criteria, test_loader):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for t_x, t_y in test_loader:
            t_x, t_y = t_x.to(device), t_y.to(device)

            output = model(t_x.view(-1, 28, 28))
            test_loss += criteria(output, t_y)
            pred = output.max(1, keepdim=True)[1]

            correct += pred.eq(t_y.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test loss:{:.4f}, accuracy:{:.3f}\n'.format(test_loss, 100 * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    # Hyper Parameters
    EPOCHS = 5
    BATCH_SIZE = 64
    INPUT_SIZE = 28
    TIME_STEP = 28
    DEVICE = ['cpu', 'cuda'][torch.cuda.is_available()]
    LR = 1e-2
    DOWNLOAD_MNIST = False

    print(f'Device: {DEVICE}')

    # download data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, ],
                             std=[0.229, ]),
    ])

    train_data = datasets.MNIST(
        root='./',
        train=True,
        download=DOWNLOAD_MNIST,
        transform=transform,
    )

    test_data = datasets.MNIST(
        root='./',
        train=False,
        download=DOWNLOAD_MNIST,
        transform=transform,
    )

    # load data
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    print(f'train_loader batch:{len(train_loader)}')

    print(test_data.data.size())
    print(test_data.targets.size())

    # plt.imshow(train_data.data[0].numpy(), cmap='gray')
    # plt.title(f'{train_data.targets[0]}')
    # plt.show()

    rnn = RNN()
    rnn.to(DEVICE)
    print(rnn)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        train_model(rnn, DEVICE, loss_func, train_loader, optimizer, epoch)
        test_model(rnn, DEVICE, loss_func, test_loader)
        scheduler.step()