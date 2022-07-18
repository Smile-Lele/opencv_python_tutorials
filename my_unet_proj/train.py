from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from data_loading import BasicDataset
# from unet_model import UNet
from unet import Unet
from dice_score import dice_loss

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 50
BATCH_SIZE = 1
LR = 1e-2


def train_model(model, train_loader, criterion, optimizer):
    model.train()
    for data in train_loader:
        inputs = data['image'].to(DEVICE)
        targets = data['mask'].to(DEVICE)

        outputs = model(inputs)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(outputs, targets)
        # loss += dice_loss(F.softmax(outputs, dim=1).float(),
        #                   F.one_hot(targets, model.classes).permute(0, 3, 1, 2).float(),
        #                   multiclass=True)
        loss.backward()
        optimizer.step()

        print(f'train: {loss.item()=}')

    return loss


def val_model(model, val_loader, criterion):
    model.eval()
    with torch.no_grad():
        for data in val_loader:
            inputs = data['image'].to(DEVICE)
            targets = data['mask'].to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss += dice_loss(F.softmax(outputs, dim=1).float(),
                              F.one_hot(targets, model.classes).permute(0, 3, 1, 2).float(),
                              multiclass=True)
            print(f'val: {loss.item()=}')

    return loss


def ready_dataset():
    dir_img = Path('./data_/dataset/imgs')
    dir_mask = Path('./data_/dataset/labels')
    val_percent: float = 0.1

    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_mask)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    loader_args = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = ready_dataset()

    # model = UNet(n_channels=3, n_classes=2).to(DEVICE)
    model = Unet(classes=2, pretrained=True).to(DEVICE)
    model.freeze_backbone()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), LR)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(EPOCHS):
        train_loss = train_model(model, train_loader, criterion, optimizer)
        val_loss = val_model(model, val_loader, criterion)
        print(f'{epoch}:{train_loss=} | {val_loss=}')
        lr_scheduler.step()
