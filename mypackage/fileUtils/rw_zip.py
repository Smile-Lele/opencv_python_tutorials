import os
import zipfile
from concurrent import futures

import pyminizip
import tqdm

from mypackage import str_utils


def unzip_helper(file, filename, dst, pwd):
    with zipfile.ZipFile(file, 'r') as f:
        f.extract(filename, dst, pwd)
    return filename


def unzip(file, dst, pwd=None):
    title = os.path.split(file)[-1]
    with zipfile.ZipFile(file, 'r') as zf:
        with futures.ProcessPoolExecutor() as executor:
            to_do_list = list()
            for member in zf.infolist():
                future = executor.submit(unzip_helper, file, member.filename, dst, pwd)
                to_do_list.append(future)
            done_iter = futures.as_completed(to_do_list)
            done_iter = tqdm.tqdm(done_iter, total=len(zf.infolist()), desc='Unzip ' + str(title))
            res = [future.result() for future in done_iter]
    return len(list(res)) == len(zf.infolist())


def zip_helper(files, dst):
    with zipfile.ZipFile(dst, 'w') as z:
        split = os.path.split
        [z.write(file, os.path.join('', split(file)[-1])) for file in files]


def zipp(files, dst, pwd=None):
    if pwd:
        pyminizip.compress_multiple(files, [], dst, pwd, 5)
    else:
        zip_helper(files, dst)


if __name__ == '__main__':
    files = str_utils.scan_files(os.getcwd(), subdir=False)
    files = str_utils.file_filter(files, types=['.py'])
    zipp(files, os.path.join(os.getcwd(), 'test.zip'))
