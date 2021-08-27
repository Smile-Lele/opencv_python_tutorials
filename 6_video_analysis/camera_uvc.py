# coding: utf-8

import cv2 as cv


def set_params(cap, fps=18, exposure=7):
    # TODO: worning fourcc must be placed at the end of params settting,
    # OPENCV BUG, source code see in:
    # https://github.com/opencv/opencv/blob/68d15fc62edad980f1ffa15ee478438335f39cc3/modules/videoio/src/cap_dshow.cpp
    ret = [cap.set(cv.CAP_PROP_EXPOSURE, exposure - 14),
           cap.set(cv.CAP_PROP_FPS, fps),
           cap.set(cv.CAP_PROP_GAMMA, 100),
           cap.set(cv.CAP_PROP_GAIN, 0),
           cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280),
           cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720),
           cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))]
    return all(ret)


def get_params(cap):
    def decode_fourcc(v):
        v = int(v)
        return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

    fourcc = decode_fourcc(cap.get(cv.CAP_PROP_FOURCC))
    fps = cap.get(cv.CAP_PROP_FPS)
    exposure = cap.get(cv.CAP_PROP_EXPOSURE) + 14  # exposure: from -13 to -1
    gamma = cap.get(cv.CAP_PROP_GAMMA)
    gain = cap.get(cv.CAP_PROP_GAIN)
    # width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    info = f'Mode:{fourcc} | FPS:{fps:.2f} | Expo:{exposure} | Gamma:{gamma} | Gain:{gain}'
    print(info)


def grab(cap, frame_nums=float('inf')):
    frame_counter = 0
    imgs = list()
    while cap.isOpened() and frame_counter < frame_nums:
        _ret, img = cap.read()
        if not _ret:
            raise ValueError('fail to capture frame!')
        if frame_counter < 1:
            get_params(cap)
            print(f'Shape:{img.shape}')

        if frame_nums < 1e10:
            imgs.append(img)
        frame_counter += 1
        cv.imshow("Video", img)
        key = cv.waitKey(1)
        if key == 27:
            break
    cv.destroyAllWindows()
    cap.release()
    print(f'Frames:{frame_counter}')
    print(f'Capture done:{frame_counter == frame_nums}')
    return imgs


def main():
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError('device not found')

    ret = set_params(cap, 120, 50)
    if not ret:
        raise ValueError('fail to set parameters')

    imgs = grab(cap)
    cap.release()


if __name__ == '__main__':
    main()
