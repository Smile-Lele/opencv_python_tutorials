import cv2 as cv


def write(key: str, data, path: str):
    file = cv.FileStorage(path, flags=cv.FILE_STORAGE_WRITE)
    file.write(key, data)


def read(path: str, key: str):
    fs = cv.FileStorage(path, flags=cv.FILE_STORAGE_FORMAT_XML)
    return fs.getNode(key).mat()
