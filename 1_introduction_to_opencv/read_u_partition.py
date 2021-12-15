import time
from datetime import datetime
import glob
import os
from shutil import copy2
from concurrent import futures
import tqdm

import psutil


def scan_partitions():
    ps = psutil.disk_partitions()
    # print(ps)
    devices = [p.mountpoint for p in ps if 'nosuid' in p.opts or 'removable' in p.opts]
    return devices


def copy_file(src, dst):
    copy2(src, dst)


def copy_files(srcpath, dstpath, filter):
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)

    files = [os.path.join(root, file) for root, subdic, files in os.walk(srcpath) for file in files if
             os.path.splitext(file)[-1] in filter]

    with futures.ThreadPoolExecutor(max_workers=len(files)) as executor:
        [executor.submit(copy_file, file, dstpath) for file in tqdm.tqdm(files, total=len(files), desc='Process')]


def main():
    while True:
        # scan u partitions
        devices = scan_partitions()
        # print(devices)

        # copy files
        for device in devices:
            dstpath = os.path.join(os.getcwd(), datetime.now().strftime("%Y-%m-%d_%H%M%S"))
            copy_files(device, dstpath, ['.xls'])

        break
        time.sleep(10)


if __name__ == '__main__':
    main()
