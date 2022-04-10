import cv2 as cv
from mypackage import icv
from matplotlib import pyplot as plt
import numpy as np

img = icv.imread_ex('mask.png', cv.IMREAD_GRAYSCALE)
row, col = img.shape[:2]

x = np.arange(0, col, 1)
y = np.arange(0, row, 1)
X, Y = np.meshgrid(x, y)


Z = img[Y, X]

fig = plt.figure('surface')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, cmap='rainbow')
# ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
