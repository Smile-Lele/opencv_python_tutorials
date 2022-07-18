import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

# HYPER PARAMETER
LR = 0.01
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 256


class ClassificationNetwork(nn.Module):
    # define your network structure
    def __init__(self):
        # 下式等价于nn.Module.__init__.(self)
        super(ClassificationNetwork, self).__init__()  # RGB 3*32*32
        self.conv1 = nn.Conv2d(3, 15, 3)  # 输入3通道，输出15通道，卷积核为3*3
        self.bn1 = nn.BatchNorm2d(15)
        self.conv2 = nn.Conv2d(15, 75, 4)  # 输入15通道，输出75通道，卷积核为4*4
        self.bn2 = nn.BatchNorm2d(75)
        self.conv3 = nn.Conv2d(75, 375, 3)  # 输入75通道，输出375通道，卷积核为3*3
        self.bn3 = nn.BatchNorm2d(375)
        self.fc1 = nn.Linear(1500, 400)  # 输入2000，输出400
        self.bn4 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 120)  # 输入400，输出120
        self.bn5 = nn.BatchNorm1d(120)
        self.fc3 = nn.Linear(120, 84)  # 输入120，输出84
        self.bn5 = nn.BatchNorm1d(84)
        self.fc4 = nn.Linear(84, 10)  # 输入 84，输出 10（分10类）

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)  # 3*32*32  -> 150*30*30  -> 15*15*15
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)  # 15*15*15 -> 75*12*12  -> 75*6*6
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)  # 75*6*6   -> 375*4*4   -> 375*2*2
        x = x.view(x.size()[0], -1)  # 将375*2*2的tensor打平成1维，1500
        x = F.relu(self.bn4(self.fc1(x)))  # 全连接层 1500 -> 400
        x = F.relu(self.fc2(x))  # 全连接层 400 -> 120
        x = F.relu(self.fc3(x))  # 全连接层 120 -> 84
        x = self.fc4(x)  # 全连接层 84  -> 10
        return x


cifar10_train = torchvision.datasets.CIFAR10("CIFAR10", train=True,
                                             transform=lambda x: np.array(x, dtype=np.float32).transpose(2, 0, 1),
                                             target_transform=None, download=True)
cifar10_test = torchvision.datasets.CIFAR10("CIFAR10", train=False,
                                            transform=lambda x: np.array(x, dtype=np.float32).transpose(2, 0, 1),
                                            target_transform=None, download=True)

net = ClassificationNetwork().to(DEVICE)
print(net)


def testset_precision(net, testset):
    net.eval()
    dl = DataLoader(testset, batch_size=128)
    total_count = 0
    total_correct = 0
    for data in dl:
        inputs = data[0].to(DEVICE)
        targets = data[1].to(DEVICE)
        outputs = net(inputs)
        predicted_labels = outputs.argmax(dim=1)
        comparison = predicted_labels == targets
        total_count += predicted_labels.size(0)
        total_correct += comparison.sum()
    net.train()

    return int(total_correct) / int(total_count)


print(f'Inital precision: {testset_precision(net, cifar10_test)}')


class DrawingBoard:
    def __init__(self, names):
        self.data = {}
        for name in names:
            self.data[name] = []

    def update(self, data_dict):
        for key in data_dict:
            self.data[key].append(data_dict[key])

    def draw(self):
        all_keys = list(self.data.keys())
        fig, ax = plt.subplots(nrows=1, ncols=len(all_keys))
        for idx in range(len(all_keys)):
            ax[idx].plot(self.data[all_keys[idx]])
            ax[idx].set_title(all_keys[idx])
        plt.show()


# modify as you need
criterion = nn.CrossEntropyLoss() # Loss Function

# 回归 / 分类

# 回归任务，使用 nn.MSELoss()
# 分类任务，使用 nn.CrossEntropyLoss()

# optimizer = optim.Adam(net.parameters(), lr=0.01)

optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)

dataloader = DataLoader(cifar10_train, batch_size=BATCH_SIZE, shuffle=True)


import time
import signal
class TimeLimitation:
    def __init__(self, limit):
        self.limit = limit

    def __enter__(self):
        def handler(signum, frame):
            raise NotImplementedError('Time\'s up')
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.limit)

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)


# with TimeLimitation(60): # don't forget the indentation
    # fill your training code

# modify as you need
print('start training')
db = DrawingBoard(['training_loss', 'testset_precision'])
for epoch in range(1, 16):
    for data in dataloader:
        inputs = data[0].to(DEVICE)
        targets = data[1].to(DEVICE)
        outputs = net(inputs)

        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # db.update({'training_loss': loss.item(), 'testset_precision': testset_precision(net, cifar10_test)})
    # db.draw()
    print(f'{epoch=} | training_loss:{loss.item()} | testset_precision:{testset_precision(net, cifar10_test)}')
    scheduler.step()

print('finished')
