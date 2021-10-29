import numpy as np
import cv2 as cv

from matplotlib import pyplot as plt

x = np.random.randint(25, 100, 25)
y = np.random.randint(175, 255, 25)
z = np.hstack((x, y))
z = z.reshape((50, 1))
z = np.float32(z)
plt.hist(z, 256, [0, 256]), plt.show()

# 定义终止标准 = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# 设置标志
flags = cv.KMEANS_RANDOM_CENTERS

# 应用K均值
compactness, labels, centers = cv.kmeans(z, 2, None, criteria, 10, flags)
print(centers)

A = z[labels == 0]
B = z[labels == 1]

# 现在绘制用红色'A'，用蓝色绘制'B'，用黄色绘制中心
plt.hist(A, 256, [0, 256], color='r')
plt.hist(B, 256, [0, 256], color='b')
plt.hist(centers, 32, [0, 256], color='y')
plt.show()
