import cv2 as cv
import numpy as np
from mypackage.imUtils import icv
from mypackage.timeUtils import timer


@timer.clock
def main():
    imdict = {}
    img = icv.imread_ex('S000057_P1.png', cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    imdict['src'] = img

    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)
    cnt = max(contours, key=cv.contourArea)
    rows, cols = img.shape[:2]
    canvas = np.zeros_like(img, dtype=np.float64)
    for y in range(rows):
        for x in range(cols//4, cols):
            dist = cv.pointPolygonTest(cnt, (x, y), measureDist=True)
            if dist < 0:
                canvas[y, x] = dist

    imdict['dist'] = canvas
    icv.implot_ex(imdict)


@timer.clock
def morph():
    img = icv.imread_ex('S000057_P1.png', cv.IMREAD_GRAYSCALE)
    # img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (13, 13))
    cv.morphologyEx(img, cv.MORPH_DILATE, kernel, iterations=120)


if __name__ == '__main__':
    main()
    # morph()