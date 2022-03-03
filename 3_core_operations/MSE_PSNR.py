import cv2 as cv
import numpy as np
from mypackage.imUtils import icv

img1 = icv.imread_ex('c5.png', cv.IMREAD_COLOR)
img2 = icv.imread_ex('c7.png', cv.IMREAD_COLOR)


def PSNRandMSE(src1, src2):
    psnr = cv.PSNR(src1, src2)
    mse = 255 ** 2 / 10 ** (psnr / 10)
    return psnr, mse


def SSIM(src1, src2):
    """
    Referance: ”Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli,
    “Image quality assessment: From error visibility to structural similarity,”
    IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, Apr. 2004.”
    """
    c1, c2 = 6.5025, 58.5225
    src1 = np.float32(src1)
    src2 = np.float32(src2)

    i1_1 = src1 * src1
    i2_2 = src2 * src2
    i1_2 = src1 * src2

    mu1 = cv.GaussianBlur(src1, (11, 11), 1.5)
    mu2 = cv.GaussianBlur(src2, (11, 11), 1.5)

    mu1_1 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_2 = mu1 * mu2

    sigma1_1 = cv.GaussianBlur(i1_1, (11, 11), 1.5) - mu1_1
    sigma2_2 = cv.GaussianBlur(i2_2, (11, 11), 1.5) - mu2_2
    sigma1_2 = cv.GaussianBlur(i1_2, (11, 11), 1.5) - mu1_2

    t1 = 2 * mu1_2 + c1
    t2 = 2 * sigma1_2 + c2
    t3 = t1 * t2

    t1 = mu1_2 + mu2_2 + c1
    t2 = sigma1_1 + sigma2_2 + c2
    t1 = t1 * t2

    ssim_map = t3 / t1

    mssim = np.mean(ssim_map)

    return mssim


if __name__ == '__main__':
    res = PSNRandMSE(img1, img2)
    print(res)

    res1 = SSIM(img1, img2)
    print(res1)
