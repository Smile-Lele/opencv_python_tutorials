# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
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
        nn.Dropout(0.5)
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
    EPOCHS = 10
    BATCH_SIZE = 64
    INPUT_SIZE = 28
    DEVICE = ['cpu', 'cuda'][torch.cuda.is_available()]
    LR = 3e-3
    DOWNLOAD_MNIST = False

    print(f'Device: {DEVICE}')

    # download data
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, ],
        #                      std=[0.229, ]),
    ])

    train_db = datasets.MNIST(
        root='./',
        train=True,
        download=DOWNLOAD_MNIST,
        transform=transform,
    )

    # split training data to train and validation
    NUM_TRAIN_DATA = train_db.data.size()[0]
    train_num = np.uint(NUM_TRAIN_DATA * 0.8)
    valid_num = np.uint(NUM_TRAIN_DATA - train_num)
    train_data, valid_data = torch.utils.data.random_split(train_db, [train_num, valid_num])

    # training data
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    # validation data
    valid_loader = DataLoader(
        dataset=valid_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    # test data
    test_data = datasets.MNIST(
        root='./',
        train=False,
        download=DOWNLOAD_MNIST,
        transform=transform,
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    # plt.imshow(train_data.data[0].numpy(), cmap='gray')
    # plt.title(f'{train_data.targets[0]}')
    # plt.show()

    rnn = RNN()
    rnn.to(DEVICE)
    # print(rnn)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)
    loss_func = nn.CrossEntropyLoss().to(DEVICE)

    for epoch in range(EPOCHS):
        train_model(rnn, DEVICE, loss_func, train_loader, optimizer, epoch)
        test_model(rnn, DEVICE, loss_func, test_loader)
        scheduler.step()