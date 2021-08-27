import cv2 as cv
import dlib
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

    # use UVC camera to capture
    # cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    # ret = [cap.set(cv.CAP_PROP_EXPOSURE, 8 - 14),
    #        cap.set(cv.CAP_PROP_FPS, 120),
    #        cap.set(cv.CAP_PROP_GAMMA, 100),
    #        cap.set(cv.CAP_PROP_GAIN, 0),
    #        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280),
    #        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720),
    #        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))]
    # if not ret:
    #     raise ValueError(f'fail to set params:{ret}')

    _, init_frame = cap.read()
    box = cv.selectROI('frame', init_frame, fromCenter=False, showCrosshair=True)
    x, y, w, h = box

    tracker = dlib.correlation_tracker()
    rect = dlib.rectangle(x, y, x + w, y + h)
    tracker.start_track(init_frame, rect)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        tracker.update(frame)
        pos = tracker.get_position()
        start_x = np.uint(pos.left())
        start_y = np.uint(pos.top())
        end_x = np.uint(pos.right())
        end_y = np.uint(pos.bottom())
        cv.rectangle(frame, (start_x, start_y), (end_x, end_y), random.choice(COLORS), 2)

        cv.imshow('frame', frame)
        key = cv.waitKey(60) & 0XFF
        if key == 27:
            break


    cv.destroyAllWindows()
    cap.release()



if __name__ == '__main__':
    main()