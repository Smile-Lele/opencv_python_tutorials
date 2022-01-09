import numpy as np


class Ndarray_:
    def __init__(self, path):
        self.path = path

    def read(self):
        with np.load(self.path) as npz:
            data = npz['data']
        return data

    def write(self, data):
        np.savez(self.path, data=data)
