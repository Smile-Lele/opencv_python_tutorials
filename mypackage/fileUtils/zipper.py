import glob
import os
import zipfile
from concurrent import futures

import pyminizip
import tqdm


def zip_(files, destPath):
    with zipfile.ZipFile(destPath, 'w') as z:
        for file in files:
            dir_, _ = os.path.split(file)
            zip_dir = file.replace(dir_, '')
            z.write(file, zip_dir)


def unzip(filePath, filename, destPath, pwd=b'heygears008'):
    with zipfile.ZipFile(filePath, 'r') as f:
        f.extract(filename, destPath, pwd)
    return filename


def decompress_ultm(filePath, destPath):
    title = os.path.split(filePath)[-1]
    with zipfile.ZipFile(filePath, 'r') as zf:
        with futures.ProcessPoolExecutor() as executor:
            to_do_list = list()
            for member in zf.infolist():
                future = executor.submit(unzip, filePath, member.filename, destPath)
                to_do_list.append(future)
            done_iter = futures.as_completed(to_do_list)
            done_iter = tqdm.tqdm(done_iter, total=len(zf.infolist()), desc='Unzip ' + str(title))
            for future in done_iter:
                res = future.result()
    return len(list(res)) == len(zf.infolist())


def compress_ultm(filePath, destPath):
    file_ext = [os.path.join(filePath, '*.png'), os.path.join(filePath, '*.ini')]
    files = glob.glob(file_ext[0])
    files += glob.glob(file_ext[1])
    zip_(files, destPath)


def compress_ultm_with_pwd(filePath, destPath, pwd=b'heygears008'):
    file_ext = [os.path.join(filePath, '*.png'), os.path.join(filePath, '*.ini')]
    files = glob.glob(file_ext[0])
    files += glob.glob(file_ext[1])
    pyminizip.compress_multiple(files, [], destPath, pwd, 5)
