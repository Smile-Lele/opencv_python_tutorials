# coding: utf-8

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from visdom import Visdom
import numpy as np

# Hyper Parameters
EPOCHS = 1000
BATCH_SIZE = 64
DEVICE = ('cpu' if torch.cuda.is_available() else 'cpu')
LR = 1e-3


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


global_step = 0


def train_model(model, device, criteria, train_loader, optimizer, epoch, viz):
    global global_step
    model.train()
    for step, sample in enumerate(train_loader):
        b_x, b_y = sample[:, 0].view(-1, 1), sample[:, 1].view(-1, 1)
        b_x, b_y = b_x.to(device), b_y.to(device)

        pred = model(b_x)
        loss = criteria(pred, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1
        viz.line([loss.item()/len(b_x)], [global_step], win='train_loss', update='append')
        # viz.scatter(X=np.column_stack((b_y.data.numpy(), pred.data.numpy())), win='origin')

        if step % 50 == 0:
            print(f'Epoch:{epoch}, Loss:{loss.item()/len(b_x) :.4f}')


def val_model(model, device, criteria, val_loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for sample in val_loader:
            t_x, t_y = sample[:, 0].view(-1, 1), sample[:, 1].view(-1, 1)
            t_x, t_y = t_x.to(device), t_y.to(device)

            pred = model(t_x)
            val_loss += criteria(pred, t_y)

    val_loss /= len(val_loader.dataset)
    print('Val Loss:{:.4f}\n'.format(val_loss))
    return val_loss


def main():
    inps = torch.unsqueeze(torch.linspace(-3, 3, 2500), dim=1)
    tgts = inps.pow(2) + 0.3 * torch.rand(inps.size())
    dataset = TensorDataset(inps, tgts)

    dataset_n = len(dataset)
    train_n = round(0.6 * dataset_n)
    val_n = round(0.3 * dataset_n)
    test_n = dataset_n - train_n - val_n
    train_set, val_set, test_set = random_split(torch.tensor(dataset), [train_n, val_n, test_n],
                                                generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set,
                              batch_size=BATCH_SIZE,
                              num_workers=2,
                              shuffle=True,
                              pin_memory=True)

    val_loader = DataLoader(val_set,
                            batch_size=BATCH_SIZE,
                            num_workers=2,
                            shuffle=True,
                            pin_memory=True)

    test_loader = DataLoader(test_set,
                             batch_size=BATCH_SIZE,
                             num_workers=2,
                             shuffle=True,
                             pin_memory=True)

    model = Model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.5)
    criteria = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

    # visdom
    viz = Visdom()
    viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
    viz.line([0.], [0.], win='val_loss', opts=dict(title='val loss'))
    colors = np.random.randint(0, 255, (1, 3))
    viz.scatter(X=np.column_stack((inps, tgts)), win='origin', opts=dict(title='origin & pred', markercolor=colors))

    for epoch in range(EPOCHS):
        train_model(model, DEVICE, criteria, train_loader, optimizer, epoch, viz)
        val_loss = val_model(model, DEVICE, criteria, val_loader)
        viz.line([val_loss], [epoch], win='val_loss', update='append')
        scheduler.step(val_loss)


if __name__ == '__main__':
    main()
