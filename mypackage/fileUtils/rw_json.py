# coding: utf-8

import json


def read(path):
    with open(path, 'r') as f:
        _data = f.read()
    return json.loads(_data)


def write(path, data):
    with open(path, 'w') as f:
        _data = json.dumps(data)
        f.write(_data)



