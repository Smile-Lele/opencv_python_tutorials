import torchvision.models as models
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


def train_model(model, device, criteria, train_loader, optimizer, epoch):
    model.train()
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x, b_y = b_x.to(device), b_y.to(device)

        output = model(b_x)
        loss = criteria(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(f'Epoch:{epoch}, Loss:{loss.item() :.6f}')


def test_model(model, device, criteria, test_loader):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for t_x, t_y in test_loader:
            t_x, t_y = t_x.to(device), t_y.to(device)

            output = model(t_x)
            test_loss += criteria(output, t_y)
            pred = output.max(1, keepdim=True)[1]

            correct += pred.eq(t_y.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test loss:{:.4f}, accuracy:{:.3f}\n'.format(test_loss, 100 * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    # Hyper Parameters
    EPOCHS = 10
    BATCH_SIZE = 32
    DEVICE = ['cpu', 'cuda'][torch.cuda.is_available()]
    LR = 1e-2
    DOWNLOAD_MNIST = True
    print(f'Device: {DEVICE}')

    transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
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

    # build model using pretrained resnet101
    resnet101 = models.resnet101(pretrained=True)
    # print(resnet101)

    # freeze all layers except fc layer
    for param in resnet101.parameters():
        param.requires_grad = False

    # set fc layer: Linear(in_features=2048, out_features=10, bias=True)
    num_ftrs = resnet101.fc.in_features
    resnet101.fc = nn.Sequential(nn.Linear(num_ftrs, 10))

    # use GPU
    resnet101 = resnet101.to(DEVICE)

    params_to_update = list()
    print(f'params to learn:')
    for name, param in resnet101.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print(f'\tname:{name}')

    # print custom resnet
    print(resnet101)

    # set optimizer
    optimizer = torch.optim.Adam(resnet101.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criteria = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        train_model(resnet101, DEVICE, criteria, train_loader, optimizer, epoch)
        test_model(resnet101, DEVICE, criteria, test_loader)
        scheduler.step()
