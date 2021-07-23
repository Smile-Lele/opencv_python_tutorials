# coding: utf-8

import cv2 as cv
import numpy as np
import random

cap = cv.VideoCapture('../mydata/drops.avi')
ret, pre_frame = cap.read()
pre_frame_gray = cv.cvtColor(pre_frame, cv.COLOR_BGR2GRAY)
prevPts = cv.goodFeaturesToTrack(pre_frame_gray, maxCorners=100, qualityLevel=0.3, minDistance=30)

mask = np.zeros_like(pre_frame)
COLORS = np.random.randint(0, 255, size=(100, 3)).tolist()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    curPts, status, _ = cv.calcOpticalFlowPyrLK(pre_frame_gray, frame_gray, prevPts, None, winSize=(15, 15), maxLevel=2)

    valid_prevPts = prevPts[status == 1]
    valid_curPts = curPts[status == 1]

    for i, (pre, cur) in enumerate(zip(valid_prevPts, valid_curPts)):
        color = random.choice(COLORS)
        mask = cv.line(mask, np.uint(pre.tolist()), np.uint(cur.tolist()), color, 2)
        frame = cv.circle(frame, np.int16(pre.tolist()), 5, color, -1)

    img = cv.add(frame, mask)
    cv.imshow('frame', img)
    if cv.waitKey(50) & 0XFF == ord('q'):
        break

    pre_frame_gray = frame_gray.copy()
    prevPts = valid_curPts.reshape(-1, 1, 2)

cv.destroyAllWindows()
cap.release()
