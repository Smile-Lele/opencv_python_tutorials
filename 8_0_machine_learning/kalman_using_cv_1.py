import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# construct object and define the dimension of matrix
kalman = cv.KalmanFilter(1, 1, cv.CV_32F)
# F
kalman.measurementMatrix = np.array([1], np.float32)
# B
kalman.controlMatrix = np.array([])
# H
kalman.transitionMatrix = np.array([1], np.float32)
# Q
kalman.processNoiseCov = np.array([1], np.float32) * 1e-3
# R
kalman.measurementNoiseCov = np.array([1], np.float32) * 2
# x_0
kalman.statePre = np.array([1], np.float32) * 35


# observe value
LENGTH = 1000
idx = list(range(LENGTH))
z = np.random.normal(60, 2, LENGTH).astype(np.float32)

plt.ion()
plt.show()

x_est = []
z_list = []
i_list = []
pred = np.array([[1]], np.float32)
for i, z_i in enumerate(z):
    # update, in opencv, update should be prior to predict
    kalman.correct(np.array([z_i]))

    # predict
    kalman.predict()

    # visualize
    x_hat = kalman.statePost[0, 0]
    x_est.append(x_hat)
    i_list.append(i)
    z_list.append(z_i)

    if i % 10 == 0:
        plt.clf()
        plt.plot(i_list, z_list, color='b')
        plt.plot(i_list, x_est, color='r')
        plt.pause(0.1)

plt.ioff()
plt.show()
