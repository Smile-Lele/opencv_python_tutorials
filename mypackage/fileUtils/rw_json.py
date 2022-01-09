# coding: utf-8

import json


class Json_:
    def __init__(self, path):
        self.path = path

    def read(self):
        with open(self.path, 'r') as f:
            _data = f.read()
        return json.loads(_data)

    def write(self, data):
        with open(self.path, 'w') as f:
            _data = json.dumps(data)
            f.write(_data)
