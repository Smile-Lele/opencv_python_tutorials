import multiprocessing
import os
from concurrent import futures
import cv2 as cv
from mypackage import icv
from mypackage import str_utils


def process(file_):
    img = icv.imread_ex(file_, cv.IMREAD_GRAYSCALE)
    dir, fname_ext, fname, _ = str_utils.split_dir(file_)
    thre, thre_img = cv.threshold(img, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    dst = os.path.join(dir, 'data')
    str_utils.check_make_dir(dst)
    icv.imwrite_ex(os.path.join(dst, fname_ext), thre_img)
    return 1


if __name__ == '__main__':
    multiprocessing.freeze_support()
    files = str_utils.scan_files(os.getcwd(), False)
    img_files = str_utils.file_filter(files, ['.png', '.jpg'])
    print(img_files)
    with futures.ProcessPoolExecutor() as executor:
        future_list = []
        for file in img_files:
            future = executor.submit(process, file)
            future_list.append(future)
        done_iter = futures.as_completed(future_list)

    print(all([it.result() for it in done_iter]))
