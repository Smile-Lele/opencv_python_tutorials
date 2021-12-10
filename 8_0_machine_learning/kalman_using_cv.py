import cv2 as cv
import numpy as np

# construct object and define the dimension of matrix
kalman = cv.KalmanFilter(4, 2, cv.CV_32F)
# F
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
# B
# kalman.controlMatrix = np.array([])
# H
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
# Q
kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * 2
# R
kalman.measurementNoiseCov = np.array([[1, 0],
                                       [0, 1]], np.float32) * 1e-3
# x_0
kalman.statePre = np.array([[0],
                            [0],
                            [0],
                            [0]], np.float32)

# to visualize
frame = np.zeros((800, 800, 3), np.uint8)


def mousemove(event, x, y, s, p):
    m_val = np.array([[x], [y]]).astype(np.float32)

    # Updates
    kalman.correct(m_val)

    # predict
    kalman.predict()

    # visualize
    m_val = m_val.astype(np.int0)
    m_val_x, m_val_y = m_val[0, 0], m_val[1, 0]

    pred_val = kalman.statePost.astype(np.int0)
    pred_val_x, pred_val_y = pred_val[0, 0], pred_val[1, 0]
    
    cv.circle(frame, (m_val_x, m_val_y), 2, (0, 0, 255), -1)
    cv.circle(frame, (pred_val_x, pred_val_y), 2, (0, 255, 0), -1)


cv.namedWindow("kalman_tracker")
cv.setMouseCallback("kalman_tracker", mousemove)

while True:
    cv.imshow("kalman_tracker", frame)
    if cv.waitKey(10) & 0xff == 27:
        break
    if cv.waitKey(10) & 0xff == ord("s"):
        cv.imwrite('kalman.jpg', frame)
        break

cv.destroyAllWindows()
