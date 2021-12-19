import os
import sqlite3

import cv2 as cv
import numpy as np

from mypackage.imUtils import icv
from mypackage.strUtils import str_utils


# read database
def query(dbfile, tablename, colname):
    conn = sqlite3.connect(dbfile)
    cur = conn.cursor()
    QUERRY = f'SELECT {colname} FROM {tablename}'
    data_ = cur.execute(QUERRY)
    # conn.close()
    return list(data_)


# update database
def update(dbfile, tablename, idname, colname, data):
    conn = sqlite3.connect(dbfile)
    cur = conn.cursor()
    cur.executemany(f"UPDATE {tablename} SET {colname} = ? WHERE {idname} = ?", data)
    conn.commit()
    # conn.close()


# insert database
def insert(dbfile, tablename, idname, colname, data: list):
    conn = sqlite3.connect(dbfile)
    cur = conn.cursor()
    cur.execute(f'DROP TABLE IF EXISTS {tablename}')
    cur.execute(f'CREATE TABLE {tablename} ({idname} INTEGER NOT NULL, {colname} INTEGER, PRIMARY KEY({idname}))')
    cur.executemany(f'INSERT INTO {tablename} VALUES(?, ?)', data)
    conn.commit()
    # conn.close()


# update database
def update_gray(imgfile, dbfile, mshape):
    mask = icv.imread_ex(imgfile, cv.IMREAD_GRAYSCALE)
    calib_mtx = icv.img_to_mat(mask, mshape)
    flat_calib_mtx = calib_mtx.reshape(1, -1).squeeze().astype(np.uint8).tolist()
    data = [[val, id] for id, val in enumerate(flat_calib_mtx, 1)]

    update(dbfile, 't_light_uniformity_adjust', 'id', 'gray', data)


def insert_gray(imgfile, dbfile, mshape):
    mask = icv.imread_ex(imgfile, cv.IMREAD_GRAYSCALE)
    calib_mtx = icv.img_to_mat(mask, mshape)
    flat_calib_mtx = calib_mtx.reshape(1, -1).squeeze().astype(np.uint8).tolist()
    data = [(id, val) for id, val in enumerate(flat_calib_mtx, 1)]
    insert(dbfile, 't_light_uniformity_adjust', 'id', 'gray', data)


if __name__ == '__main__':
    try:
        mshape = np.int0(str_utils.user_input_number('Mshape'))

        imgfiles = str_utils.read_multifiles(os.getcwd(), 'png')
        dbfiles = str_utils.read_multifiles(os.getcwd(), 'db')

        # update_gray(imgfiles[0], dbfiles[0], mshape[:2])
        insert_gray(imgfiles[0], dbfiles[0], mshape[:2])

        data = query(dbfiles[0], colname='gray', tablename='t_light_uniformity_adjust')
        calib_mtx = np.asarray(data).astype(np.uint8).reshape(mshape[0], mshape[1])
        print(calib_mtx)
    except Exception as e:
        print(e)
    finally:
        os.system('pause')
