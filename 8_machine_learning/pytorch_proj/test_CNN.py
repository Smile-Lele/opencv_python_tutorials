
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(  # (1, 28, 28)
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,  # (3-1)/2 = 1, (k - 1) / 2 = p
            ),  # (32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (32, 14, 14)
        )

        self.conv2 = nn.Sequential(  # (32, 14, 14)
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (64, 7, 7)
        )

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output


def train_model(model, device, train_loader, optimizer, epoch):
    model.train()
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x, b_y = b_x.to(device), b_y.to(device)

        output = model(b_x)
        loss = nn.CrossEntropyLoss()(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(f'Epoch:{epoch}, Loss:{loss.item() :.6f}')


def test_model(model, device, test_loader):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for t_x, t_y in test_loader:
            t_x, t_y = t_x.to(device), t_y.to(device)

            output = model(t_x)
            test_loss += nn.CrossEntropyLoss()(output, t_y)
            pred = output.max(1, keepdim=True)[1]

            correct += pred.eq(t_y.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test loss:{:.4f}, accuracy:{:.3f}\n'.format(test_loss, 100 * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    # Hyper Parameters
    EPOCHS = 10
    BATCH_SIZE = 16
    LR = 0.001  # learning rate
    DOWNLOAD_MNIST = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'DEVICE:{DEVICE}')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))
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

    plt.imshow(test_data.data[0].numpy(), cmap='gray')
    plt.title(f'{test_data.targets[0]}')
    plt.show()

    cnn = CNN().to(DEVICE)
    print(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        train_model(cnn, DEVICE, train_loader, optimizer, epoch)
        test_model(cnn, DEVICE, test_loader)
