import cv2 as cv
import numpy as np


def solveHelper(coeffs, func) -> list:
    return func(coeffs)[1].squeeze().tolist()


def solve_ex(coeffs: list, y):
    assert len(coeffs) != 0, f'Error {len(coeffs)=}'
    coeffs = np.float32(coeffs)
    coeffs[-1] -= y
    return solveHelper(coeffs, cv.solveCubic) if len(coeffs) == 4 else solveHelper(coeffs[::-1], cv.solvePoly)


def filterRoots(roots: list, lo, hi):
    assert len(roots) != 0, f'Error: {len(roots)=}'
    if isinstance(roots[0], list) and len(roots[0]) == 2:
        print(roots)
        return [r[0] for r in roots if lo < r[0] < hi and abs(r[1]) < 1e-8]
    return list(filter(lambda r: lo < r < hi, roots))


if __name__ == '__main__':
    Kcubic = [1.5562457571961154e-05, -0.0030010165293595133, 0.36700608560942793, -2.2864298375913563]
    Kpoly = [-2.5784741814209814e-09, 1.371098452598373e-06, -0.000265186669975475, 0.024553759043765444,
             -0.9246907111037501, 20.80802584389039]

    roots = solve_ex(Kcubic, 20)
    print(filterRoots(roots, 60, 180))
