import numpy as np
import cv2 as cv
from mypackage.multiplot import multiplot as mplt

imdict = dict()
img = cv.imread('../mydata/messi5.jpg', cv.IMREAD_COLOR)
assert img is not None, 'img should be not empty'
imdict['imdict'] = img

denoise_img = cv.fastNlMeansDenoisingColored(img, None, 5, 7, 21)
imdict['denoise_im'] = denoise_img

gauss = cv.GaussianBlur(denoise_img, (5, 5), 0)
imdict['gauss'] = gauss

edges = cv.Canny(img, 100, 200, L2gradient=True)
imdict['edges'] = edges

mplt.show(imdict)
