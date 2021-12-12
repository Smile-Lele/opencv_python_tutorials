import glob
import os
import sqlite3
import cv2 as cv
from mypackage.imUtils import icv
import numpy as np
from mypackage.strUtils import str_utils


# read database
def query(dbfile, tablename, colname):
    conn = sqlite3.connect(dbfile)
    cur = conn.cursor()
    QUERRY = f'SELECT {colname} FROM {tablename}'
    data_ = cur.execute(QUERRY)
    return list(data_)


# update database
def update(dbfile, tablename, colname, idname, data):
    conn = sqlite3.connect(dbfile)
    cur = conn.cursor()
    cur.executemany(f"UPDATE {tablename} SET {colname} = ? WHERE {idname} = ?", data)
    conn.commit()


# update database
def update_gray(imgfile, dbfile, mshape):
    mask = icv.imread_ex(imgfile, cv.IMREAD_GRAYSCALE)
    calib_mtx = icv.img_to_mat(mask, mshape)
    flat_calib_mtx = calib_mtx.reshape(1, -1).squeeze().astype(np.uint8).tolist()
    data = [[val, id] for id, val in enumerate(flat_calib_mtx, 1)]

    update(dbfile, 't_light_uniformity_adjust', 'gray', 'id', data)


if __name__ == '__main__':
    mshape = np.int0(str_utils.user_input_number('Mshape'))

    imgfiles = str_utils.read_multifiles(os.getcwd(), 'png')
    dbfiles = str_utils.read_multifiles(os.getcwd(), 'db')

    update_gray(imgfiles[0], dbfiles[0], mshape[:2])

    data = query(dbfiles[0], colname='gray', tablename='t_light_uniformity_adjust')
    calib_mtx = np.asarray(data).astype(np.uint8).reshape(4, 6)
    print(calib_mtx)

    os.system('pause')
