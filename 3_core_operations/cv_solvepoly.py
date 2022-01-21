import cv2 as cv
import numpy as np


def solvePoly1d_ex(coeffs):
    """
    solveCubic:
    coeffs[0]x3+coeffs[1]x2+coeffs[2]x+coeffs[3]=0

    solvePoly:
    coeffs[n]xn+coeffs[n−1]xn−1+...+coeffs[1]x+coeffs[0]=0

    """
    length = len(coeffs)
    assert length != 0, f'Error: {coeffs=}'
    coeffs = np.asarray(coeffs)

    def func(y):
        coeffs[-1] -= y
        if length == 4:
            return cv.solveCubic(coeffs)[1].reshape(-1, 1)

        coeffs[::] = coeffs[::-1]
        return cv.solvePoly(coeffs)[1].reshape(-1, 2)

    return func


def clipRoots(roots, minLmt=0, maxLmt=255):
    if roots.shape[1] == 1:
        idx = np.where((roots > minLmt) & (roots < maxLmt))
        return roots[idx]

    res = [row[0] for row in roots if abs(row[1]) < 1e-8 and minLmt < row[0] < maxLmt]
    return res


K = [-2.2894486573926043e-09, 1.2951682162058233e-06, -0.0002562838071868683, 0.023833612568105716, -0.8276905920566396, 19.80463980106381]
# K = [2.427464014933865e-05, -0.005119180264153617, 0.5875008587749303, -6.253427396552328]
f = solvePoly1d_ex(K)

a = f(18)

res = clipRoots(a, 0, 180)
print(res)

# idx = np.where((a[:, 0] > 60) & (a[:, 0] < 180) & (a[:, 1] < 1e-16))
# print(a[idx])
