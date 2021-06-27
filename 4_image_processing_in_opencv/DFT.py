import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from mypackage.multiplot import multiplot as mplt

imdict = dict()
img = cv.imread('../mydata/captured_white.png', 0)
imdict['img'] = img

rows, cols = img.shape
print(img.shape)

nrows = cv.getOptimalDFTSize(rows)
ncols = cv.getOptimalDFTSize(cols)

right = ncols - cols
bottom = nrows - rows
nimg = cv.copyMakeBorder(img, 0, bottom, 0, right, cv.BORDER_CONSTANT, value=0)
print(nimg.shape)

dft = cv.dft(np.float32(nimg), flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
dft_ch1, dft_ch2 = cv.split(dft_shift)
magnitude_spectrum = 20 * np.log(cv.magnitude(dft_ch1, dft_ch2))
imdict['magnitude_spectrum'] = magnitude_spectrum

c_r = nrows // 2
c_c = ncols // 2

mask = np.ones((nrows, ncols, 2), dtype=nimg.dtype)
mask = cv.circle(mask, (c_c, c_r), 100, (0, 0), -1, cv.LINE_8)

# apply mask and inverse DFT
fshift = dft_shift * mask
fshift[fshift == 0] = 0.001
f_ch1, f_ch2 = cv.split(fshift)
magnitude_spectrum_masked = 20 * np.log(cv.magnitude(f_ch1, f_ch2))
imdict['magnitude_spectrum_masked'] = magnitude_spectrum_masked

f_ishift = np.fft.ifftshift(fshift)
img_back = cv.idft(f_ishift)
img_back_ch1, img_back_ch2 = cv.split(img_back)
img_back = cv.magnitude(img_back_ch1, img_back_ch2)
imdict['img_back'] = img_back

mplt.show(imdict)
