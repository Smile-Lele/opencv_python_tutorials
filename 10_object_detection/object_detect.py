import cv2 as cv
import numpy as np
import random


COLORS = np.random.randint(0, 255, size=(100, 3)).tolist()


def main():
    # load the path of images
    cv.samples.addSamplesDataSearchPath('../mydata')
    file = cv.samples.findFile('drops.avi')
    if not file:
        raise FileNotFoundError('file not found')

    cap = cv.VideoCapture(file)
    trackers = cv.MultiTracker_create()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        state, boxes = trackers.update(frame)

        for box in boxes:
            x, y, w, h = box.astype(np.uint)
            cv.rectangle(frame, (x, y), (x + w, y + h), random.choice(COLORS), 2)

        cv.imshow('frame', frame)
        key = cv.waitKey(50) & 0XFF

        if key == ord('s'):
            box = cv.selectROI('frame', frame, fromCenter=False, showCrosshair=True)
            tracker = cv.TrackerKCF_create()
            trackers.add(tracker, frame, box)

        elif key == 27:
            break
    cv.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
