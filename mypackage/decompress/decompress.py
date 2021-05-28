import os
import time
import zipfile
from concurrent import futures

import tqdm


def extract(filepath, filename, destpath, pwd):
    with open(filepath, 'rb') as f:
        zf = zipfile.ZipFile(f)
        zf.extract(filename, destpath, pwd)
    return filename


def decompress(filepath, destpath, pwd=b'heygears008'):
    with open(filepath, 'rb') as f:
        zf = zipfile.ZipFile(f)
        with futures.ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            to_do_list = list()
            for member in zf.infolist():
                future = executor.submit(extract, filepath, member.filename, destpath, pwd)
                to_do_list.append(future)
            done_iter = futures.as_completed(to_do_list)
            done_iter = tqdm.tqdm(done_iter, total=len(zf.infolist()), desc='Decompress')
            for future in done_iter:
                res = future.result()
    return len(list(res)) == len(zf.infolist())


if __name__ == '__main__':
    start_time = time.time()
    file_path = 'D:/MyData/Model/'
    file_name = '05-jituo.ultm'
    dest_path = file_path + '/File/'
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)

    decompress(os.path.join(file_path, file_name), dest_path)
    print(time.time() - start_time)
