import numpy as np
import matplotlib.pyplot as plt

# Hyper Parameter
Q = np.mat([0.0001])  # process noise covariance
R = np.mat([4])  # observe noise covariance

# Initialize p and x
x = np.mat([0])
p = np.mat([1])

# state transfer matrix
F = np.mat([1])
H = np.mat([1])
B = np.array([1])

# observe value
idx = list(range(1000))
z = np.random.normal(60, 2, 1000)
u = 0

plt.ion()
plt.show()

x_est = []
z_list = []
i_list = []
for i, z_i in enumerate(z):
    # predict
    x_hat = F * x + B * u
    p_hat = F * p * F.T + Q

    # update
    K = p_hat * H.T / (H * p_hat * H.T + R)
    x = x_hat + K * (z_i - H * x_hat)
    p = (np.eye(H.shape[1]) - K * H) * p_hat
    x_est.append(x[0, 0])
    i_list.append(i)
    z_list.append(z_i)

    if i % 5 == 0:
        plt.clf()
        plt.plot(i_list, z_list, color='b')
        plt.plot(i_list, x_est, color='r')
        plt.pause(0.1)

plt.ioff()
plt.show()
