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
    class PROP(Enum):
        CAP_PROP_FRAME = 6601

    def __init__(self, device_idx, api=cv.CAP_ANY):
        self.device_idx = device_idx
        self.api = api
        self.frames = []
        self.cap = cv.VideoCapture(self.device_idx, self.api)
        print(Fore.CYAN + f'CAM={self.cap.isOpened()}')
        print(Fore.CYAN + f'API={self.cap.getBackendName()}')
        self.cap_props = {eval('cv.' + cap_prop): cap_prop for cap_prop in dir(cv) if cap_prop.startswith("CAP_PROP_")}

    def set(self, prop_id, value):
        if prop_id == self.PROP.CAP_PROP_FRAME:
            ret = self.cap.set(3, value[0]) and self.cap.set(4, value[1])
            if not ret or abs(self.cap.get(3) - value[0]) > 1e-3 or abs(self.cap.get(4) - value[1]) > 1e-3:
                print(Fore.RED + f'SET_CAP_PROP_FRAME:{self.cap.get(3)}x{self.cap.get(4)}')
                return ret
            print(Fore.GREEN + f'CAP_PROP_FRAME={self.cap.get(3)}x{self.cap.get(4)}')
            return ret

        ret = self.cap.set(prop_id, value)
        tgt = self.cap.get(prop_id)
        if prop_id == cv.CAP_PROP_FOURCC:
            tgt = self.decode_fourcc(tgt)

        if not ret or abs(self.cap.get(prop_id) - value) > 1e-2:
            print(Fore.RED + f'SET_{self.cap_props.get(prop_id)}:{tgt}')
            return False
        print(Fore.GREEN + f'{self.cap_props.get(prop_id)}={tgt}')
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

    def decode_fourcc(self, v):
        return ''.join([chr((int(v) >> 8 * i) & 0xFF) for i in range(4)])

    def encode_fourcc(self, v: str):
        return cv.VideoWriter_fourcc(*v)

    def release(self):
        self.cap.release()


if __name__ == '__main__':
    cam = CamReader(0, cv.CAP_DSHOW)
    cam.set(cam.PROP.CAP_PROP_FRAME, (1280, 720))
    cam.set(cv.CAP_PROP_FPS, 10)
    cam.set(cv.CAP_PROP_EXPOSURE, -6)
    cam.set(cv.CAP_PROP_GAMMA, 100),
    cam.set(cv.CAP_PROP_GAIN, 0),
    cam.set(cv.CAP_PROP_FOURCC, cam.encode_fourcc('YUY2'))
    while cam.isOpened():
        frame = cam.read()
        # cam.record('test.mp4')
        frame = cv.rectangle(frame, (50, 50), (1280-50, 720-50), (0, 0, 255), 3, cv.LINE_AA)
        cv.imshow('frame', frame)
        k = cv.waitKey(5)
        if k == 27:
            cv.imwrite('test.bmp', frame)
            cv.destroyWindow('frame')
            break

    # cam.capture(60)
    # img = cam.concat()
    # cv.imshow('f', img)
    # cv.waitKey()
