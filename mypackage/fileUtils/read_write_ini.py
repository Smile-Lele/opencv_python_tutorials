import numpy as np


def write(path, data):
    with open(path, 'w') as f:
        f.writelines(data)


def read(path):
    with open(path, 'r') as f:
        _data = f.readlines()
    return _data


if __name__ == '__main__':
    data = read('buildscript.ini')
    print(len(data))
    rand_ = np.arange(1, 274).tolist()
    print(rand_)
    
    print(list(zip(data, rand_)))
    new_data = [row.replace('\n', str(new_item) + ', \n') for row, new_item in zip(data, rand_)]
    print(new_data)
    
    write('buildscript.ini', new_data)
