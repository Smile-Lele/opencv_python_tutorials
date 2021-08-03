import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn import decomposition

# create random data
mean = [20, 20]  # 均值
cov = [[5, 0], [25, 25]]  # 协方差矩阵
X = np.random.multivariate_normal(mean, cov, 500)
x, y = X.T

# calculate PCA
mean = np.empty(0)
mean, eig, eigval = cv.PCACompute2(X, mean, 0.99)  # 99% of variance is retained
print(f'mean:{mean}')
print(f'eig:{eig}')
print(f'eigv:{eigval}')
mean_ = mean.squeeze()

plt.plot(x, y, 'o', zorder=1)
plt.quiver(mean_[0], mean_[1], eig[0, 0], eig[0, 1], zorder=3, scale=0.2, units='xy')
plt.quiver(mean_[0], mean_[1], eig[1, 0], eig[1, 1], zorder=3, scale=0.2, units='xy')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

#
Z = cv.PCAProject(X, mean, eig)
plt.plot(Z[:, 0], Z[:, 1], 'o')
plt.show()


X_B = cv.PCABackProject(Z, mean, eig)
plt.plot(X_B[:, 0], X_B[:, 1], 'o')
plt.show()

# # 实现独立主成分分析      基于sklearn
# ica = decomposition.FastICA()
# X3 = ica.fit_transform(X)
# plt.plot(X3[:, 0], X3[:, 1], 'o')
# plt.xlabel('first independent component')
# plt.ylabel('second independent component')
# # plt.savefig('ica.png')
# plt.show()  # sklearn 提供的快速ICA分析

