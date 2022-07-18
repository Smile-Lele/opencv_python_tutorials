from mypackage.imUtils import icv
from matplotlib import pyplot as plt

if __name__ == '__main__':

    pnts = [(x, x) for x in range(1, 50, 1)]

    coeffs = icv.polyfit(pnts, 3)

    print(coeffs)

    for p in pnts:
        print(p[0], p[1])
        print(p[0], icv.polyfunc(p[0], coeffs))

