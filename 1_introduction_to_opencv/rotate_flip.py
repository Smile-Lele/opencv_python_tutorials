import cv2 as cv
from mypackage import icv

imdict = {}
img = cv.imread('S000084_P1.png', cv.IMREAD_GRAYSCALE)
imdict['img'] = img

flip_img = cv.flip(img, flipCode=-1)
imdict['flip'] = flip_img

rot_img = cv.rotate(img, cv.ROTATE_180)
imdict['rot'] = rot_img

diff = flip_img - rot_img
imdict['diff'] = diff

icv.implot_ex(imdict)


"""
In conclusion, simultaneous horizontal and vertical flipping of the image
is equal to rotation of 180 degrees.
"""