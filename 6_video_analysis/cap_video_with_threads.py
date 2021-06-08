# coding: utf-8
import time

import cv2 as cv
import numpy as np
import os
from concurrent import futures

import tqdm as tqdm


def capture_video(filename):
    cap = cv.VideoCapture(filename)
    frames = list()
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        frames.append(frame)
    return cap, frames


def write_video(cap, images, despath):
    fourcc = int(cap.get(cv.CAP_PROP_FOURCC))
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_size = (np.int0(cap.get(cv.CAP_PROP_FRAME_WIDTH)), np.int0(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

    if not os.path.exists(despath):
        os.mkdir(despath)

    writer = cv.VideoWriter(os.path.join(despath, 'output.avi'), fourcc, fps, frame_size)

    try:
        for image in images:
            writer.write(image)
    finally:
        writer.release()
        cap.release()
        cv.destroyAllWindows()


def process_frame(index, image):
    image = cv.fastNlMeansDenoising(image, None, 5, 7, 21)
    image = cv.GaussianBlur(image, (5, 5), 0)
    return index, image


def process_video(images):
    to_do_list = list()
    MAX_WORKERS = len(images)
    with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for idx, image in enumerate(images):
            future = executor.submit(process_frame, idx, image)
            to_do_list.append(future)

        done_iter = futures.as_completed(to_do_list)
        done_iter = tqdm.tqdm(done_iter, total=len(images), desc='Process')

        res = [future.result() for future in done_iter]
        sort_res = sorted(res, key=lambda r: r[0])
        tar_images = [res[1] for res in sort_res]
    return tar_images


if __name__ == '__main__':
    filename = os.path.join('../mydata', 'drops.avi')

    cap, images = capture_video(filename)

    tar_images = process_video(images)

    filepath = os.path.split(filename)[0]
    despath = os.path.join(filepath, 'res')
    write_video(cap, tar_images, despath)
