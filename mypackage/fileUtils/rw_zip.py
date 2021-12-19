import glob
import os
import zipfile
from concurrent import futures

import pyminizip
import tqdm


def unzip_helper(file, filename, dst, pwd=b'heygears008'):
    with zipfile.ZipFile(file, 'r') as f:
        f.extract(filename, dst, pwd)
    return filename


def unzip(file, dst):
    title = os.path.split(file)[-1]
    with zipfile.ZipFile(file, 'r') as zf:
        with futures.ProcessPoolExecutor() as executor:
            to_do_list = list()
            for member in zf.infolist():
                future = executor.submit(unzip_helper, file, member.filename, dst)
                to_do_list.append(future)
            done_iter = futures.as_completed(to_do_list)
            done_iter = tqdm.tqdm(done_iter, total=len(zf.infolist()), desc='Unzip ' + str(title))
            res = [future.result() for future in done_iter]
    return len(list(res)) == len(zf.infolist())


def zip_helper(files, dst):
    with zipfile.ZipFile(dst, 'w') as z:
        for file in files:
            dir_, _ = os.path.split(file)
            zip_dir = file.replace(dir_, '')
            z.write(file, zip_dir)


def zip_(file, dst, filter, pwd=b'heygears008'):
    files = [os.path.join(file, name) for name in os.listdir(file)
             if os.path.splitext(name)[-1] in filter and os.path.isfile(os.path.join(file, name))]
    if pwd:
        pyminizip.compress_multiple(files, [], dst, pwd, 5)
        return
    zip_helper(files, dst)
