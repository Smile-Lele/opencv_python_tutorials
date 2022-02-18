import cv2 as cv
from mypackage.imUtils import icv
from mypackage.timeUtils import timer


def read_img(filename):
    return icv.imread_ex(filename, cv.IMREAD_GRAYSCALE)


@timer.clock
def big_kernel(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (21, 21))
    cv.morphologyEx(img, cv.MORPH_DILATE, kernel, iterations=100)


@timer.clock
def small_kernel(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    cv.morphologyEx(img, cv.MORPH_DILATE, kernel, iterations=100)


if __name__ == '__main__':
    img = read_img('S000071_P1.png')
    # icv.imshow_ex(img)

    big_kernel(img)
    small_kernel(img)