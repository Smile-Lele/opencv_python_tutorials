# coding: utf-8

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# Hyper Parameters
LR = 1e-3
BATCH_SIZE = 16
EPOCH = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PATH = ''


# Define Model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def train(model, train_data, optimizer, criterion):
    model.train()  # set training mode
    for x, y in train_data:
        optimizer.zero_grad()  # clean gradient
        x, y = x.to(DEVICE), y.to(DEVICE)  # move data to GPU
        pred = model(x)  # compute output
        loss = criterion(pred, y)  # compute loss
        loss.backward()  # compute gradient back propagation
        optimizer.step()  # update net parameter using new gradient


def validate(model, valid_data, criterion):
    model.eval()  # set evaluation mode
    total_loss = 0
    for x, y in valid_data:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            pred = model(x)
            loss = criterion(pred, y)

        total_loss += loss.cpu().item() * len(x)
        avg_loss = total_loss / len(valid_data.dataset)
    return avg_loss


def test(model, test_data):
    model.eval()
    preds = []
    for x in test_data:
        x = x.to(DEVICE)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.cpu())
    return preds


def main():
    # prepare data
    train_dataset = Dataset()
    train_data = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    valid_dataset = Dataset()
    valid_data = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    test_dataset = Dataset()
    test_data = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # load model
    model = Model().to(DEVICE)

    # define loss
    criterion = nn.MSELoss()

    # select optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=False)

    # training model
    for epoch in range(EPOCH):
        train(model, train_data, optimizer, criterion)
        avg_loss = validate(model, valid_data, criterion)
        scheduler.step(avg_loss)

    # test model
    pred = test(model, test_data)

    # save
    torch.save(model.state_dict(), PATH)

    # load
    ckpt = torch.load(PATH)
    model.load_state_dict(ckpt)
