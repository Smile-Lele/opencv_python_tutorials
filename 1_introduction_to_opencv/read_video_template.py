# coding: utf-8

import cv2 as cv

# load the path of images
cv.samples.addSamplesDataSearchPath('../mydata')
file = cv.samples.findFile('vtest.avi')
if not file:
    raise FileNotFoundError('file not found')

cap = cv.VideoCapture(file)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv.imshow('frame2', frame)
    k = cv.waitKey(15) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        ...
cv.destroyAllWindows()
cap.release()