import torch
import torch.optim as optim

print('cuda={}'.format(torch.cuda.is_available()))
print(dir(optim))

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)
shuffled_indices = torch.randperm(n_samples)
train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

print(train_indices)
print(val_indices)

train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]

val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]

train_t_un = 0.1 * train_t_u
val_t_un = 0.1 * val_t_u


def model(t_u, w, b):
    return w * t_u + b


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()


"""
dummy version
"""
# def training_loop(n_epochs, optimizer, params, train_t_u, val_t_u, train_t_c, val_t_c):
#     for epoch in range(1, n_epochs + 1):
#         train_t_p = model(train_t_u, *params)
#         train_loss = loss_fn(train_t_p, train_t_c)
#
#         val_t_p = model(val_t_u, *params)
#         val_loss = loss_fn(val_t_p, val_t_c)
#
#         optimizer.zero_grad()
#         train_loss.backward()  # 注意没有val_loss.backward因为不能在验证集上训练模型
#         optimizer.step()
#
#         if epoch <= 3 or epoch % 500 == 0:
#             print('Epoch %d, Training loss %.2f, Validation loss %.2f' % (
#                 epoch, float(train_loss), float(val_loss)))
#     return params


"""
optimized version 1
"""

# def training_loop(n_epochs, optimizer, params,
#                   train_t_u, val_t_u, train_t_c, val_t_c):
#     for epoch in range(1, n_epochs + 1):
#         train_t_p = model(train_t_u, *params)
#         train_loss = loss_fn(train_t_p, train_t_c)
#
#         with torch.no_grad():
#             val_t_p = model(val_t_u, *params)
#             val_loss = loss_fn(val_t_p, val_t_c)
#             assert val_loss.requires_grad == False
#
#         optimizer.zero_grad()
#         train_loss.backward()
#         optimizer.step()


"""
optimized verson 2
"""


def calc_forward(t_u, t_c, params, is_train):
    with torch.set_grad_enabled(is_train):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
    return loss


def training_loop(n_epochs, optimizer, params, train_t_u, val_t_u, train_t_c, val_t_c):
    for epoch in range(1, n_epochs + 1):
        train_loss = calc_forward(train_t_u, train_t_c, params, True)
        val_loss = calc_forward(val_t_u, val_t_c, params, False)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch <= 3 or epoch % 500 == 0:
            print('Epoch %d, Training loss %.2f, Validation loss %.2f' % (epoch, float(train_loss), float(val_loss)))


params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)

training_loop(
    n_epochs=3000,
    optimizer=optimizer,
    params=params,
    train_t_u=train_t_un,
    val_t_u=val_t_un,
    train_t_c=train_t_c,
    val_t_c=val_t_c)


"""
线性模型是用于拟合数据的合理的最简单的模型；
凸优化技术可以用于线性模型，但不能推广到神经网络，因此本章重点介绍参数估计。
深度学习可用于通用模型，这些通用模型不是为解决特定任务而设计的，而是可以自动调整以专门解决眼前的问题。
学习算法等于根据观察结果优化模型的参数。损失函数是对执行任务中的错误的一种度量，例如预测输出值和测量值之间的误差。目标就是使损失函数值尽可能低。
损失函数关于模型参数的变化率可用于在减少损失的方向上更新该参数。
PyTorch中的optim模块提供了一组现成的优化器，用于更新参数和最小化损失函数。
优化器使用PyTorch的autograd来计算每个参数的梯度，而梯度具体取决于该参数对最终输出的贡献程度。autograd允许用户在复杂的前向通过过程中依赖于动态计算图。
诸如torch.no_grad()的上下文管理器可用于控制是否需要自动求导。
数据通常划分为独立的训练集和验证集，从而可以在未训练的数据（即验证集）上进行模型评估。
当模型的性能在训练集上继续提高但在验证集上下降时，模型就发生了过拟合。这种情况通常发生在模型无法泛化（到训练集之外的数据）而是记住了训练集所需的输出时。
"""