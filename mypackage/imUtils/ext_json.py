# coding: utf-8

import json

import numpy as np


def write(path, data):
    with open(path, 'w') as f:
        if isinstance(data, (list, dict)):
            _data = json.dumps(data)
            f.write(_data)
        else:
            raise TypeError('data should be list()')
    return True


def read(path):
    with open(path, 'r') as f:
        _data = f.read()
    return json.loads(_data)


def read_func(row, col, variable='z'):
    func_dict = read('./_data/func_data.json')
    key = '_'.join(['f', str(row), str(col)])
    return func_dict[key]


def read_cam_para(jsonfile='./_data/cam_para.json'):
    para_dict = read(jsonfile)
    return para_dict['mtx'], para_dict['dist']


def read_energy_to_mat(mshape, jsonfile='./_data/EnergyCalibParams.json'):
    data = read(jsonfile)
    gray = data['Gray']
    value = gray['Value']
    row, col = mshape
    num = 0
    mat = np.zeros((row, col), np.uint8)
    for r in range(row):
        for c in range(col):
            mat[r, c] = value[num]
            num += 1
    return mat


def read_roi_param(jsonfile='./_data/roiparam.json'):
    param = read(jsonfile)
    return param['x'], param['y'], param['w'], param['h']


def modify_energy(energy_mat, jsonfile='./_data/EnergyCalibParams.json'):
    data = read(jsonfile)
    assert 'Gray' in data.keys(), 'Gray should be existed'
    gray = data['Gray']
    gray['Value'] = energy_mat
    write(jsonfile, data)
    print('writing done')
