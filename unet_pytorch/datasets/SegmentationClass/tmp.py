import cv2

from mypackage import icv

img = icv.imread_ex('1.png', cv2.IMREAD_UNCHANGED)
imdict = {}
imdict['img']= img
icv.implot_ex(imdict)