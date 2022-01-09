from . import rw_ini
from . import rw_json
from . import rw_ndarray


class IFile:
    def __init__(self, path: str):
        self.path = path

        if path.endswith('.ini'):
            self.rw_obj = rw_ini.Ini_(self.path)

        if path.endswith('.json'):
            self.rw_obj = rw_json.Json_(self.path)

        if path.endswith('.npz'):
            self.rw_obj = rw_ndarray.Ndarray_(self.path)

    def read(self):
        print(f'{self.path} is read')
        return self.rw_obj.read()

    def write(self, data):
        print(f'{self.path} is writen')
        self.rw_obj.write(data)
