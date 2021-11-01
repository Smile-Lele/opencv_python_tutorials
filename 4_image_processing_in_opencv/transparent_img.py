import cv2 as cv
import numpy as np


def otsu_threshold(img, min_thre=0, max_thre=255, offset=0, inv=False, visibility=False):
    img = cv.GaussianBlur(img, (3, 3), 0)

    min_thre += offset
    max_thre -= offset
    img[img < min_thre] = min_thre
    img[img > max_thre] = max_thre

    thre, thre_img = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print(f'otsu thre:{thre}')

    if inv:
        thre_img = cv.bitwise_not(thre_img)

    if visibility:
        temp = cv.resize(thre_img, None, fx=0.3, fy=0.3, interpolation=cv.INTER_AREA)
        cv.imshow('otsu_thre', temp)
        cv.waitKey()
        cv.destroyWindow('otsu_thre')
    return thre, thre_img


img = cv.imread('IMG_5804.JPG', cv.IMREAD_GRAYSCALE)
thre, img = otsu_threshold(img)


img_BGRA = cv.cvtColor(img, cv.COLOR_GRAY2BGRA)
b, g, r, alpha = cv.split(img_BGRA)

alpha[img >= thre] = 0
print(alpha.min())

img_BGRA = cv.merge((b, g, r, alpha))
cv.imshow('', img_BGRA)
cv.imwrite('esign.png', img_BGRA)
cv.waitKey()
