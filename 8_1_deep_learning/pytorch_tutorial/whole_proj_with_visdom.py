import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

from visdom import Visdom

batch_size = 200
LR = 3e-3
epochs = 10

train_data = datasets.MNIST('../data', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                # transforms.Normalize((0.1307,), (0.3081,))
                            ]))

# split training data to train and validation
NUM_TRAIN_DATA = train_data.data.size()[0]
train_num = np.uint(NUM_TRAIN_DATA * 0.8)
valid_num = np.uint(NUM_TRAIN_DATA - train_num)
train_db, valid_db = torch.utils.data.random_split(train_data, [train_num, valid_num])

# training for train model
train_loader = torch.utils.data.DataLoader(
    train_db,
    batch_size=batch_size, shuffle=True)

# validation for avoiding overfitting and selecting optimal accuracy
valid_loader = torch.utils.data.DataLoader(
    valid_db,
    batch_size=batch_size, shuffle=True)

# only for testing model
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)

        return x


device = torch.device('cuda:0')
net = MLP()
net.to(device)
print(list(net.children()))
optimizer = optim.Adam(net.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss().to(device)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min')

viz = Visdom()

viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
viz.line([[0.0, 0.0]], [0.], win='test', opts=dict(title='test loss&acc.',
                                                   legend=['loss', 'acc.']))
global_step = 0

for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.cuda()

        logits = net(data)
        loss = criterion(logits, target)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()

        global_step += 1
        viz.line([loss.item()], [global_step], win='train_loss', update='append')

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    # validation set for avoiding overfitting will stop early the process of training
    # validation set to select optimal parameters
    test_loss = 0
    correct = 0
    for data, target in valid_loader:
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.cuda()
        logits = net(data)
        test_loss += criterion(logits, target).item()

        pred = logits.argmax(dim=1)
        correct += pred.eq(target).float().sum().item()

    # visdom module for Visualization
    viz.line([[test_loss, correct / len(valid_loader.dataset)]],
             [global_step], win='test', update='append')
    viz.images(data.view(-1, 1, 28, 28), win='x')
    viz.text(str(pred.detach().cpu().numpy()), win='pred',
             opts=dict(title='pred'))

    test_loss /= len(valid_loader.dataset)
    scheduler.step(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
