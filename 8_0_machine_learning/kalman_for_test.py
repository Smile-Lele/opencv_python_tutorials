import cv2 as cv
import numpy as np

measurements = []
predictions = []

# 创建800*800的空帧
frame = np.zeros((800, 800, 3), np.uint8)

# 初始化测量数组和鼠标运动预测数组
last_measurement = current_measurement = np.array((2, 1), np.float32)
last_prediction = current_prediction = np.zeros((2, 1), np.float32)


# 鼠标移动回调函数，绘制跟踪结果
def mousemove(event, x, y, s, p):
    global frame, current_measurement, measurements, last_measurement, current_prediction, last_prediction
    last_prediction = current_prediction
    last_measurement = current_measurement
    current_measurement = np.array([[np.float32(x)], [np.float32(y)]])

    # Updates the predicted state from the measurement.
    kalman.correct(current_measurement)

    # Computes a predicted state.
    current_prediction = kalman.predict()

    lmx, lmy = np.int0(last_measurement[0])[0], np.int0(last_measurement[1])[0]
    cmx, cmy = np.int0(current_measurement[0])[0], np.int0(current_measurement[1])[0]
    lpx, lpy = np.int0(last_prediction[0])[0], np.int0(last_prediction[1])[0]
    cpx, cpy = np.int0(current_prediction[0])[0], np.int0(current_prediction[1])[0]

    # cv.line(frame, (lmx, lmy), (cmx, cmy), (0, 255, 0), 5)
    # cv.line(frame, (lpx, lpy), (cpx, cpy), (0, 0, 255), 5)
    cv.circle(frame, (cmx, cmy), 5, (0, 0, 255), -1)
    cv.circle(frame, (cpx, cpy), 5, (0, 255, 0), -1)


cv.namedWindow("kalman_tracker")
cv.setMouseCallback("kalman_tracker", mousemove)

'''
cv.KalmanFilter([dynamParams, measureParams[, controlParams[, type]]]) → <KalmanFilter object>

Parameters: 
    dynamParams – Dimensionality of the state.
    measureParams – Dimensionality of the measurement.
    controlParams – Dimensionality of the control vector.
    type – Type of the created matrices that should be CV_32F or CV_64F.

The class implements a standard Kalman filter http://en.wikipedia.org/wiki/Kalman_filter, [Welch95]. However, you can modify transitionMatrix, controlMatrix, and measurementMatrix to get an extended Kalman filter functionality.
See the OpenCV sample kalman.cpp .
'''
# 创建卡尔曼滤波器
kalman = cv.KalmanFilter(4, 2, 1)

kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

while True:
    cv.imshow("kalman_tracker", frame)
    if cv.waitKey(100) & 0xff == ord("q"):
        break
    if cv.waitKey(100) & 0xff == ord("s"):
        cv.imwrite('kalman.jpg', frame)
        break

cv.destroyAllWindows()
