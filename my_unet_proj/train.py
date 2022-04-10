import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from Unet import Unet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 50
BATCH_SIZE = 2
LR = 1e-4


def train_model(model, train_loader, device, criteria, optimizer):
    model.train()
    for sample in train_loader:
        b_x, b_y = sample[:, 0].view(-1, 1), sample[:, 1].view(-1, 1)
        b_x, b_y = b_x.to(device), b_y.to(device)

        pred = model(b_x)
        loss = criteria(pred, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss


def val_model(model, val_loader, device, criteria):
    model.eval()
    with torch.no_grad():
        for sample in val_loader:
            t_x, t_y = sample[:, 0].view(-1, 1), sample[:, 1].view(-1, 1)
            t_x, t_y = t_x.to(device), t_y.to(device)

            pred = model(t_x)
            loss = criteria(pred, t_y)

    return loss


if __name__ == "__main__":
    num_classes = 2
    input_shape = [512, 512]

    model = Unet(num_classes=num_classes, pretrained=True)
    model.freeze_backbone(True)

    print(model)
    exit()

    optimizer = optim.Adam(model.parameters(), LR)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)
    criteria = torch.nn.CrossEntropyLoss()

    t_u = []
    n_samples = t_u.shape[0]
    n_val = int(0.2 * n_samples)
    shuffled_indices = torch.randperm(n_samples)
    train_indices = shuffled_indices[:-n_val]
    val_indices = shuffled_indices[-n_val:]

    train_set = []
    val_set = []

    train_loader = DataLoader(train_set,
                              batch_size=BATCH_SIZE,
                              num_workers=2,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)

    val_loader = DataLoader(val_set,
                            batch_size=BATCH_SIZE,
                            num_workers=2,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)

    for epoch in range(EPOCHS):
        train_loss = train_model(model, train_loader, DEVICE, criteria, optimizer)
        val_loss = val_model(model, val_loader, DEVICE, criteria)
        print(f'{epoch}:{train_loss=} | {val_loss}')
        lr_scheduler.step()
