import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


class Kalman(cv.KalmanFilter):
    def __init__(self):
        super().__init__()
        # construct object and define the dimension of matrix
        self.kalman = cv.KalmanFilter(1, 1, cv.CV_32F)
        # F
        self.kalman.measurementMatrix = np.array([1], np.float32)
        # B
        self.kalman.controlMatrix = np.array([])
        # H
        self.kalman.transitionMatrix = np.array([1], np.float32)
        # Q
        self.kalman.processNoiseCov = np.array([1], np.float32) * 1e-3
        # R
        self.kalman.measurementNoiseCov = np.array([1], np.float32) * 0.3
        # x_0
        self.kalman.statePre = np.array([1], np.float32) * 0

        self.statepost = self.kalman.statePost[0, 0]

    def kalman_predict(self, z):
        # update, in opencv, update should be prior to predict
        self.kalman.correct(np.array([z]))

        # predict
        self.kalman.predict()

        # visualization
        self.statepost = self.kalman.statePost[0, 0]


kal = Kalman()
# observe value
LENGTH = 1000
# idx = list(range(LENGTH))
z = np.random.normal(60, 2, LENGTH).astype(np.float32)

plt.ion()
plt.show()

x_est = []
z_list = []
i_list = []
for i, z_i in enumerate(z):
    kal.kalman_predict(z_i)

    # visualize
    x_hat = kal.statepost
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