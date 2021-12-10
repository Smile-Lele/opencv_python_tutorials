import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pandas.core.frame import DataFrame

# matrix X: (3 * 500)
#    |  1  |  2  |    |  m
# x1 | x1_1, x1_2, ..., x1_m
# x2 | x2_1, x2_2, ..., x2_m
# x3 | x3_1, x3_2, ..., x3_m

# create random data
M = 500
x1 = np.random.normal(0, 14.6, 500)
x2 = np.random.normal(0, 11.68, 500)
x3 = x2 * 3 + np.random.normal(0, 1.6, 500)

data = np.vstack((x1, x2, x3))

# visualize data
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x1, x2, x3, cmap='b')
plt.show()

# normalization
row_mean = np.mean(data, axis=1)
print('mean:')
print(row_mean)
row_std = np.std(data, axis=1)

X = data.T - row_mean
X = X.T

# covariance
# C = X @ X.T / (499-1)
C = np.cov(X)
# print('C:')
# print(C)

# eigvector, eigvalue
# method1: eig, which has to be a square array
eigvalue, eigvector = np.linalg.eig(C)
print('eigvalue:')
print(eigvalue.T)
print('eigvector:')
print(eigvector.T, end='\n\n')

# method2: svd, which not necessarily is a square array (recommend)
eigvector, eigvalue, vh = np.linalg.svd(C, full_matrices=True)
print('eigvalue:')
# eigvalue_ = eigvector.T @ C @ eigvector
print(eigvalue.T)
print('eigvector:')
print(eigvector.T)

# Dimension Reduction
# eigvalue:
# [21.63865446 33.96055518  0.22084919]
# pick up first 2 rows
P = eigvector.T[:2, :]
print('P:')
print(P, end='\n\n')

pca_res = P @ data
print(DataFrame(pca_res.T))

x_1, x_2 = np.vsplit(pca_res, 2)
plt.scatter(x_1, x_2, c='b')
# plt.show()

# method3: pca using Opencv
mean = np.empty(0)
mean, eigvector, eigval = cv.PCACompute2(data.T, mean, 0.99)  # 99% of variance is retained
print('mean:')
print(mean)
print('eigval:')
print(eigval)
print('eigvector:')
print(eigvector, end='\n\n')

# method4: sklearn to implement pca
pca = PCA(n_components=2)  # n_components can be integer or float in (0,1)
pca.fit(data.T)  # fit the model
print('variance_ratio:')
print(pca.explained_variance_ratio_)
print('variance:')
print(pca.explained_variance_)
pca_res1 = pca.fit_transform(data.T)
plt.scatter(-pca_res1[:, 0], -pca_res1[:, 1], c='r')
plt.show()
print(DataFrame(pca_res1))
