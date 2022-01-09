import numpy as np


class Ini_:
    def __init__(self, path):
        self.path = path

    def write(self, data):
        with open(self.path, 'w') as f:
            f.writelines(data)

    def read(self):
        with open(self.path, 'r') as f:
            _data = f.readlines()
        return _data


if __name__ == '__main__':
    ini = Ini('buildscript.ini')
    data = ini.read()
    print(len(data))
    rand_ = np.arange(1, 274).tolist()
    print(rand_)

    print(list(zip(data, rand_)))
    new_data = [row.replace('\n', str(new_item) + ', \n') for row, new_item in zip(data, rand_)]
    print(new_data)

    ini.write(new_data)
