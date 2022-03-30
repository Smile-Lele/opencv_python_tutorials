import cv2 as cv
from enum import Enum
from colorama import Fore, Back, Style
import numpy as np


class CamWriter:
    def __init__(self, video_path, fourcc, fps, size):
        self.video = cv.VideoWriter(video_path, fourcc, fps, size)

    def record(self, frame):
        self.video.write(frame)

    def release(self):
        self.video.release()


class CamReader:
    def __init__(self, device_idx, api=cv.CAP_ANY):
        self.device_idx = device_idx
        self.api = api
        self.cap = cv.VideoCapture(self.device_idx, self.api)
        print(Fore.CYAN + f'API={self.cap.getBackendName()}')
        self.cap_props = {eval('cv.' + cap_prop): cap_prop for cap_prop in dir(cv) if cap_prop.startswith("CAP_PROP_")}
        self.frames = []

    def set(self, prop_id, value):
        ret = self.cap.set(prop_id, value)
        if not ret or self.cap.get(prop_id) != value:
            print(Fore.RED + f'SET_{self.cap_props.get(prop_id)}_ERROR:{self.cap.get(prop_id)}')
            return False
        print(Fore.GREEN + f'{self.cap_props.get(prop_id)}={value}')
        return True

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return False
        return frame

    def capture(self, n):
        for i in range(n):
            self.frames.append(self.read())
        self.release()

    def concat(self):
        grays__ = list(map(lambda x: cv.cvtColor(x, cv.COLOR_BGR2GRAY), self.frames))
        mean__ = np.mean(np.dstack(grays__), axis=2)
        return cv.convertScaleAbs(mean__)

    def isOpened(self):
        return self.cap.isOpened()

    def decode_fourcc(self, v: str):
        return ''.join([chr((int(v) >> 8 * i) & 0xFF) for i in range(4)])

    def encode_fourcc(self, v):
        return cv.VideoWriter_fourcc(*v)

    def release(self):
        self.cap.release()


if __name__ == '__main__':
    cam = CamReader(0, cv.CAP_ANY)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    cam.set(cv.CAP_PROP_FPS, 30)
    cam.set(cv.CAP_PROP_EXPOSURE, 100)
    cam.set(cv.CAP_PROP_GAMMA, 100),
    cam.set(cv.CAP_PROP_GAIN, 0),
    cam.set(cv.CAP_PROP_FOURCC, cam.encode_fourcc('XVID'))

    # while cam.isOpened():
    #     ret, frame = cam.read()
    #     # cam.record('test.mp4')
    #     cv.imshow('frame', frame)
    #     k = cv.waitKey(5)
    #     if k == 27:
    #         cv.destroyWindow('frame')
    #         break

    cam.capture(60)
    img = cam.concat()
    cv.imshow('f', img)
    cv.waitKey()
