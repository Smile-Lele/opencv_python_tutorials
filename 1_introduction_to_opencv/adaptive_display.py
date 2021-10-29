import cv2 as cv


def resize_for_display(img, screen_size=(1920*4/5, 1080*4/5)):
    row, col = img.shape[:2]
    screen_c, screen_r = screen_size
    row_ratio = screen_r / row
    col_ratio = screen_c / col
    scale_ratio = row_ratio if row_ratio <= col_ratio else col_ratio
    if scale_ratio == 1:
        return img
    else:
        INTERPOLATION = cv.INTER_AREA if scale_ratio < 1 else cv.INTER_CUBIC
        img = cv.resize(img, None, fx=scale_ratio, fy=scale_ratio, interpolation=INTERPOLATION)
        return img


if __name__ == '__main__':
    img = cv.imread('S000109_P1.png', cv.IMREAD_GRAYSCALE)
    tmp = resize_for_display(img)
    cv.imshow('', tmp)
    cv.waitKey()