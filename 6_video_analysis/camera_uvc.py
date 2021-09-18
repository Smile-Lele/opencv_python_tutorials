# coding: utf-8

import cv2 as cv


def set_params(cap, fps=60, exposure=6):
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


def grab(cap, isflip=False):
    frame_counter = 0
    save_num = 0
    show_param_enable = True
    while cap.isOpened():
        _ret, frame = cap.read()
        if not _ret:
            raise ValueError('fail to capture frame!')
        if show_param_enable:
            show_param_enable = not show_param_enable
            get_params(cap)
            print(f'Shape:{frame.shape}')

        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if isflip:
            frame = cv.flip(frame, flipCode=1)

        frame_counter += 1
        key = cv.waitKey(15)
        if key == 27:
            break
        if key == ord('s'):
            save_num += 1
            cv.imwrite(str(save_num) + '.png', frame)
            cv.waitKey(3)
            print(f'{save_num}.png captured')

        fps = int(cap.get(cv.CAP_PROP_FPS))
        font_neibour = cv.mean(frame[20:60, 15:155])[0]
        # cv.rectangle(frame, (15, 20), (155, 60), 255)
        font_color = 0
        if font_neibour < 150:
            font_color = 255
        cv.putText(frame, "FPS:{}".format(fps), (15, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, font_color, 2)
        cv.imshow("Video", frame)

    print(f'Frames:{frame_counter}')


def main():
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError('device not found')

    ret = set_params(cap, 60, 7)
    if not ret:
        raise ValueError('fail to set parameters')

    cv.namedWindow("Video")

    fps = int(cap.get(cv.CAP_PROP_FPS))
    exposure = int(cap.get(cv.CAP_PROP_EXPOSURE)) + 14  # exposure: from -13 to -1
    cv.createTrackbar("FPS", "Video", fps, 120,
                      lambda v: set_params(cap, fps=v, exposure=int(cap.get(cv.CAP_PROP_EXPOSURE)) + 14))
    cv.createTrackbar("Exposure", "Video", exposure, 13,
                      lambda v: set_params(cap, fps=int(cap.get(cv.CAP_PROP_FPS)), exposure=v))

    grab(cap)
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
