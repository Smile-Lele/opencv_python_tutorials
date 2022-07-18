import os

import cv2 as cv
import numpy as np
from concurrent import futures

from mypackage.strUtils import str_utils
from mypackage.imUtils import icv
import tqdm


def imshow_(src, delay=5):
    cv.imshow('viz', src)
    key = cv.waitKey(delay) & 0xFF
    if key == 27:
        cv.destroyWindow('viz')
        return 'ESC'
    return


def conv(src, kernel):
    src = icv.cvtBGR2Gray(src)
    dst = cv.filter2D(src, cv.CV_32FC1, kernel)
    dst = cv.convertScaleAbs(dst)
    _, dst = icv.otsuThreshold(dst, visibility=False)
    return dst


def remove_black_bar(src):
    kernel_x = np.array([[3, 0, -3],
                         [3, 0, -3],
                         [3, 0, -3]])
    kernel_y = kernel_x.T
    dy = conv(src, kernel_y)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
    dy = cv.morphologyEx(dy, cv.MORPH_DILATE, kernel, iterations=1)

    lines = cv.HoughLinesP(dy, 1, np.pi / 180, src.shape[1] // 2, src.shape[1] * 2 // 3, src.shape[0] // 3)
    lines = lines.squeeze()
    top_line, bottom_line = [], []
    for x1, y1, x2, y2 in lines:
        # cv.line(src, (x1, y1), (x2, y2), (0, 0, 255), 1)
        y = (y1 + y2) / 2
        if y < src.shape[0] // 2:
            top_line.append(y)
        else:
            bottom_line.append(y)

    offset = np.int0(src.shape[0] * 0.01)
    crop_top = np.mean(top_line).astype(np.int0) + offset if top_line else 0
    crop_bottom = np.mean(bottom_line).astype(np.int0) - offset if bottom_line else src.shape[0]

    _, src = cv.threshold(src, 128, 255, cv.THRESH_BINARY_INV)

    src[:crop_top, :] = 0
    src[crop_bottom:, :] = 0

    src[src > 0] = 1

    return src


def reshape_(src, size):
    row, col = src.shape[:2]
    diff = abs(row - col)
    half = diff // 2
    another = diff - half

    if row < col:
        src = np.pad(src, ((half, another), (0, 0), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
    elif row > col:
        src = np.pad(src, ((0, 0), (half, another), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))

    dst = icv.resize_ex(src, size)

    return dst


def process(video_name):
    frames = read_video(video_name)

    # frames = [remove_black_bar(img) for img in frames]

    frames = [reshape_(f, (512, 512)) for f in frames]

    dir, fname_ext, fname, ext = str_utils.split_dir(video_name)
    rets = [icv.imstore(os.path.join(dir, 'dataset'), fname + '_' + str(i) + '.png', img) for i, img in
            enumerate(frames)]
    return all(rets)


def process_with_multi(dir):
    files = str_utils.read_multifiles(dir, 'avi')

    # files = list(filter(lambda f: os.path.split(f)[-1].endswith('3.avi'), files))

    future_list = []
    with futures.ProcessPoolExecutor(os.cpu_count() - 2) as executor:
        for file in files:
            future = executor.submit(process, file)
            future_list.append(future)
        done_iter = futures.as_completed(future_list)
        done_iter = tqdm.tqdm(done_iter, total=len(files), desc='Process')

    return [it.result() for it in done_iter]


def read_video(video_name):
    cap = cv.VideoCapture(video_name)
    if not cap.isOpened():
        print(f'{cap.isOpened()=}')
        return

    frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
    if not frame_count:
        print(f'CAP_PROP_FRAME_COUNT')
        return

    frames = []
    for i in range(int(frame_count)):
        ret, frame = cap.read()
        if not ret:
            print(f'READ_FAILED')
            return
        frames.append(frame)
        # ret = imshow(frame, 10)
        # if ret == 'ESC':
        #     print(f'KEY_EXIT')
        #     break
    return frames


def read_multi_images():
    files = str_utils.read_multifiles('./data_/dataset/imgs', 'png')
    files = list(filter(lambda f: os.path.split(f)[-1].startswith('1_'), files))
    for f in files:
        img = cv.imread(f, cv.IMREAD_UNCHANGED)

        _, img = cv.threshold(img, 128, 255, cv.THRESH_BINARY_INV)

        ret = imshow_(img, 100)
        if ret == 'ESC':
            print(f'KEY_EXIT')
            break


if __name__ == '__main__':
    process_with_multi('./data_/src_data')
    # read_multi_images()
