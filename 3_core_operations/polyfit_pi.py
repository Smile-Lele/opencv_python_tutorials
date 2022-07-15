from mypackage.imUtils import icv
from matplotlib import pyplot as plt
from mypackage.fileUtils import myfile
import numpy as np


def read_pnts():
    file = myfile.IFile('pi.json')
    data = file.read()
    pnts = [(d['Energy'], d['Currents']) for d in data['Table']]
    return pnts


if __name__ == '__main__':
    pnts = read_pnts()
    coeffs = icv.polyfit(pnts, 3)
    pnts = np.asarray(pnts)

    print(icv.polyfunc(27, coeffs))

    plt.scatter(pnts[:, 0], pnts[:, 1], color='blue')
    plt.plot(pnts[:, 0], icv.polyfunc(pnts[:, 0], coeffs), color='red')

    plt.show()
