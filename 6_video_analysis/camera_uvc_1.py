# coding: utf-8
import os
import numpy as np
import multiprocessing
import cv2 as cv


def set_params(cap, fps=60, exposure=-6):
    # TODO: worning fourcc must be placed in the end of params settting, coz OPENCV bug
    ret = [cap.set(cv.CAP_PROP_EXPOSURE, exposure),
           cap.set(cv.CAP_PROP_FPS, fps),
           cap.set(cv.CAP_PROP_GAMMA, 100),
           cap.set(cv.CAP_PROP_GAIN, 0),
           cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280),
           cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720),
           cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'YUY2'))]
    if not all(ret):
        raise ValueError('fail to set parameters')


def get_params(cap):
    def decode_fourcc(v):
        v = int(v)
        return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

    fourcc = decode_fourcc(cap.get(cv.CAP_PROP_FOURCC))
    fps = cap.get(cv.CAP_PROP_FPS)
    exposure = cap.get(cv.CAP_PROP_EXPOSURE) + 14  # exposure: from -13 to -1
    gamma = cap.get(cv.CAP_PROP_GAMMA)
    gain = cap.get(cv.CAP_PROP_GAIN)
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    info = f'Mode:{fourcc} | SIZE:({width},{height}) | FPS:{fps:.2f} | Expo:{exposure} | Gamma:{gamma} | Gain:{gain}'
    print(info)


def grab(cap, retFrames=0, record=False, visualized=True):
    """
    This method is to capture frame from VideoCapture
    :param visualized:
    :param record:
    :param cap:
    :param retFrames: default 0, it represents images captured will not be returned
    :return: depends on frame_nums
    """

    # initialize video writer
    if record:
        fourcc = int(cap.get(cv.CAP_PROP_FOURCC))
        fps = cap.get(cv.CAP_PROP_FPS)
        frame_size = (np.int0(cap.get(cv.CAP_PROP_FRAME_WIDTH)), np.int0(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
        video_writer = cv.VideoWriter(os.path.join('output.avi'), fourcc, fps, frame_size)

    # capture frame from VideoCapture
    if not cap.isOpened():
        raise RuntimeError('device not found')

    imgs = []
    save_num = 0

    while cap.isOpened():
        _ret, img = cap.read()

        if not _ret:
            raise ValueError('fail to capture frame!')

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        fps = int(cap.get(cv.CAP_PROP_FPS))
        font_neibour = cv.mean(img[20:60, 15:155])[0]
        # cv.rectangle(frame, (15, 20), (155, 60), 255)
        font_color = 0
        if font_neibour < 150:
            font_color = 255
        cv.putText(img, "FPS:{}".format(fps), (15, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, font_color, 2)

        if retFrames:
            imgs.append(img)

            if visualized:
                cv.imshow("cap", img)

            if len(imgs) == retFrames:
                cv.destroyAllWindows()
                cap.release()
                break
        else:
            cv.imshow("cap", img)

        if record:
            video_writer.write(img)

        key = cv.waitKey(1) & 0xFF

        # key 'esc' represents exit
        if key == 27:
            cv.destroyAllWindows()
            cap.release()
            break

        # key 's; represents save
        if key == ord('s'):
            save_num += 1
            cv.imwrite(str(save_num) + '.png', img)
            cv.waitKey(3)
            print(f'{save_num}.png captured')
    else:
        cap.release()
    print(f'retFrames:{len(imgs)}')
    return imgs


def main():
    device_id = int(input('device='))
    cap = cv.VideoCapture(device_id, cv.CAP_DSHOW)
    if cap is None:
        raise NotImplementedError()
    set_params(cap, 60, -6)
    get_params(cap)
    imgs = grab(cap, 0, record=True, visualized=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
