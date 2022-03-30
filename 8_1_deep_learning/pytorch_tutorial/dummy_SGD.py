import torch

print('cuda={}'.format(torch.cuda.is_available()))

"""
机器学习就是参数估计
"""

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)


def model(t_u, w, b):
    return w * t_u + b


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()


# w = torch.ones(1)
# b = torch.zeros(1)
# t_p = model(t_u, w, b)
# print(t_p)
#
# loss = loss_fn(t_p, t_c)
# print(loss)


def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c)
    return dsq_diffs


def dmodel_dw(t_u, w, b):
    return t_u


def dmodel_db(t_u, w, b):
    return 1.0


def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dw = dloss_fn(t_p, t_c) * dmodel_dw(t_u, w, b)
    dloss_db = dloss_fn(t_p, t_c) * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.mean(), dloss_db.mean()])


def training_loop(n_epochs, learning_rate, params, t_u, t_c, print_params=True, verbose=1):
    for epoch in range(1, n_epochs + 1):
        w, b = params

        t_p = model(t_u, w, b)  # 前向传播
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w, b)  # 反向传播

        params = params - learning_rate * grad

        if epoch % verbose == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
            if print_params:
                print('    Params: ', params)
                print('    Grad  : ', grad)
    return params


t_un = t_u * 0.1
params = training_loop(
    n_epochs=5000,
    learning_rate=1e-2,
    params=torch.tensor([1.0, 0.0]),
    t_u=t_un,
    t_c=t_c,
    print_params=False,
    verbose=500)
print(params)

from matplotlib import pyplot as plt

t_p = model(t_un, *params)  # 记住你是在规范后数据上训练的

fig = plt.figure(dpi=100)
plt.xlabel("Fahrenheit")
plt.ylabel("Celsius")

plt.plot(t_u.numpy(), t_p.detach().numpy())  # 在原数据上作图
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.show()