import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from matplotlib import pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # torch requires 2 dim
y = x.pow(5) + 0 * torch.rand(x.size())

x, y = Variable(x), Variable(y)


# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.leaky_relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(1, 10, 1)
print(net)

plt.ion()
plt.show()

optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=0.001)
loss_func = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

for t in range(10000):
    prediction = net(x)

    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, f'Loss={loss.data:.4f}', fontdict={'size':'12', 'color':'red'})
        plt.pause(0.1)
        scheduler.step(loss.data)

plt.ioff()
plt.show()