import cv2 as cv
import dlib
import numpy as np
import random
from mypackage.timeUtils import FPS
from multiprocessing import Queue, Process, freeze_support


COLORS = np.random.randint(0, 255, size=(100, 3)).tolist()
fps = FPS()
fps.start()


def tracking(frame, box, inqueue, outqueue):
    """
    The demo is to use multi. processes to track objects,
    the app. creates a process for each tracker corresponding to the object
    :param frame:
    :param box:
    :param inqueue:
    :param outqueue: process cannot return value, queue is required to use
    :return:
    """
    tracker = dlib.correlation_tracker()
    x, y, w, h = box
    rect = dlib.rectangle(x, y, x + w, y + h)
    tracker.start_track(frame, rect)

    while True:
        frame = inqueue.get()

        if frame is not None:
            tracker.update(frame)
            pos = tracker.get_position()
            outqueue.put(pos)


def main():
    # in order to pack executed app, when needed to load
    freeze_support()

    # load the path of images
    cv.samples.addSamplesDataSearchPath('../mydata')
    file = cv.samples.findFile('output.avi')
    if not file:
        raise FileNotFoundError('file not found')

    # create capture object
    cap = cv.VideoCapture(file)
    _, init_frame = cap.read()

    # simulate auto detection, to be replaced after achieved by CNN
    boxes = list()
    for i in range(3):
        box = cv.selectROI('frame', init_frame, fromCenter=False, showCrosshair=True)
        x, y, w, h = box
        cv.rectangle(init_frame, (x, y), (x + w, y + h), random.choice(COLORS), 2)
        boxes.append(box)

    in_Queues = list()
    out_Queues = list()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not in_Queues:
            for box in boxes:
                in_queue = Queue()
                out_queue = Queue()
                in_Queues.append(in_queue)
                out_Queues.append(out_queue)

                p = Process(target=tracking, args=(frame, box, in_queue, out_queue))
                p.daemon = True  # daemon process is used to terminate the process, when main thread is stopped
                p.start()

                x, y, w, h = box
                cv.rectangle(frame, (x, y), (x + w, y + h), random.choice(COLORS), 2)
        else:
            for in_q in in_Queues:
                in_q.put(frame)

            for out_q in out_Queues:
                pos = out_q.get()
                start_x = np.uint(pos.left())
                start_y = np.uint(pos.top())
                end_x = np.uint(pos.right())
                end_y = np.uint(pos.bottom())
                # print((start_x, start_y), (end_x, end_y))
                cv.rectangle(frame, (start_x, start_y), (end_x, end_y), random.choice(COLORS), 2)

        cv.imshow('frame', frame)
        key = cv.waitKey(1) & 0XFF
        if key == 27:
            break

        fps.update()

    fps.stop()

    cv.destroyAllWindows()
    cap.release()

    print(f'elapsed time:{fps.elapsed()}')
    print(f'approx. FPS:{fps.fps()}')



if __name__ == '__main__':
    main()