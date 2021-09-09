# coding: utf-8

import json


def write(path, data):
    with open(path, 'w') as f:
        if isinstance(data, (list, dict)):
            _data = json.dumps(data)
            f.write(_data)
        else:
            raise TypeError('data should be list or dict')
    return True


def read(path):
    with open(path, 'r') as f:
        _data = f.read()
    return json.loads(_data)
