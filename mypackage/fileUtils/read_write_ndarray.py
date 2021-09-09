import numpy as np


def write_ndarray(file, d):
    np.savez(file, data=d)


def read_ndarray(file):
    with np.load(file) as npz:
        data = npz['data']

    return data
