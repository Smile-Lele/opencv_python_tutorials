import glob
import os
import sqlite3
import cv2 as cv
from mypackage.imUtils import imconverter as imcvt
import numpy as np
from mypackage.strUtils import str_utils


# read database
def query(dbfile):
    conn = sqlite3.connect(dbfile)
    cur = conn.cursor()
    QUERRY = 'SELECT gray FROM t_light_uniformity_adjust'
    data = cur.execute(QUERRY)
    calib_mtx = np.asarray(list(data)).astype(np.uint8).reshape(4, 6)
    print(calib_mtx)


# update database
def update(imgfile, dbfile, mshape):
    mask = cv.imdecode(np.fromfile(imgfile, dtype=np.uint8), cv.IMREAD_GRAYSCALE)

    calib_mtx = imcvt.img_to_mat(mask, mshape)
    flat_calib_mtx = calib_mtx.reshape(1, -1).squeeze().astype(np.uint8).tolist()
    data = [[val, id] for id, val in enumerate(flat_calib_mtx, 1)]

    conn = sqlite3.connect(dbfile)
    cur = conn.cursor()
    cur.executemany("UPDATE t_light_uniformity_adjust "
                    "SET gray = ?"
                    "WHERE id = ?",
                    data)
    conn.commit()


if __name__ == '__main__':
    mshape = np.int0(str_utils.user_input_number('Mshape'))

    imgfiles = glob.glob(os.path.join(os.getcwd(), '*.png'))
    dbfiles = glob.glob(os.path.join(os.getcwd(), '*.db'))

    assert len(imgfiles) != 0, f'fail to find png'
    assert len(dbfiles) != 0, f'fail to find db'

    update(imgfiles[0], dbfiles[0], mshape[:2])
    query(dbfiles[0])
    os.system('pause')