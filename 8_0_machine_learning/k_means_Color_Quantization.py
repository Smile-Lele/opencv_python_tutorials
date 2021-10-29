import numpy as np
import cv2 as cv


# load the path of images
cv.samples.addSamplesDataSearchPath('../mydata')
file = cv.samples.findFile('home.jpg')
if not file:
    raise FileNotFoundError('file not found')

img = cv.imread(file)
Z = img.reshape((-1, 3))

# 将数据转化为np.float32
Z = np.float32(Z)
# 定义终止标准 聚类数并应用k均值
criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 1.0)
K = 2
ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
# 现在将数据转化为uint8, 并绘制原图像

center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape(img.shape)
res2 = np.vstack((img, res2))
cv.imshow('res2', res2)
cv.waitKey(0)
cv.destroyAllWindows()
