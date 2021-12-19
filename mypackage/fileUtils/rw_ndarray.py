import numpy as np


def read(file):
    with np.load(file) as npz:
        data = npz['data']

    return data


def write(file, data):
    np.savez(file, data=data)
