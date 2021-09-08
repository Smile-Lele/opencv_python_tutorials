import glob
import os
import time
import zipfile
from concurrent import futures
import multiprocessing

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
    multiprocessing.freeze_support()
    start_time = time.time()
    file_path = os.getcwd()

    ultm_files = glob.glob(os.path.join(file_path, '*.ultm'))
    if not ultm_files:
        raise FileNotFoundError("Error: *.ultm cannot be found...")

    for ultm_file in ultm_files:
        file_name = os.path.split(ultm_file)[-1]  # to get file name, such as 'shuwen-4K-5X.ultm'
        file_folder = file_name.replace(".ultm", "")
        dest_path = os.path.join(file_path, file_folder)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        decompress(os.path.join(file_path, file_name), dest_path)
    print(time.time() - start_time)
