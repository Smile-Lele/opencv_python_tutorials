import cv2 as cv
import numpy as np
from mypackage.imUtils import icv
from mypackage.strUtils import str_utils

files = str_utils.read_multifiles('../data/', 'png')
images = [icv.imread_ex(file, cv.IMREAD_GRAYSCALE) for file in files]

images_np = np.dstack(images)
print(images_np.shape)

imdict = {}

images_exp = np.mean(images_np, axis=2)
imdict['exp'] = images_exp

images_var = np.var(images_np, axis=2)
imdict['var'] = images_var

images_std = np.std(images_np, axis=2)
imdict['std'] = images_std

icv.implot_ex(imdict)

print(images_var.max())
print(images_std.max())

# YUYV
# (720, 1280, 1000)
# 803.4901189999999
# 28.345901273376366

# MJPG
# (720, 1280, 1000)
# 1216.311536
# 34.87565821601078

