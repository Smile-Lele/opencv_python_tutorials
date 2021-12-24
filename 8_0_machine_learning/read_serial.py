# coding:utf-8
import re
import struct
import time

import cv2 as cv
import numpy as np
import serial
from crcmod import crcmod


class Kalman:
    def __init__(self):
        super().__init__()
        # construct object and define the dimension of matrix
        self.kalman = cv.KalmanFilter(1, 1, 0, cv.CV_32F)
        # F
        self.kalman.measurementMatrix = np.array([1], np.float32)
        # B
        self.kalman.controlMatrix = np.array([])
        # H
        self.kalman.transitionMatrix = np.array([1], np.float32)
        # Q
        self.kalman.processNoiseCov = np.array([1], np.float32) * 1e-4
        # R
        self.kalman.measurementNoiseCov = np.array([1], np.float32) * 1e-3
        # x_0
        self.kalman.statePre = np.array([1], np.float32) * 0

        self.statepost = self.kalman.statePost[0, 0]

    def kalman_predict(self, z):
        # update, in opencv, update should be prior to predict
        self.kalman.correct(np.float32(np.array([z])))

        # predict
        self.kalman.predict()

        # visualization
        self.statepost = self.kalman.statePost[0, 0]


def main(port, baudrate=9600):
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baudrate
    ser.stopbits = serial.STOPBITS_ONE
    ser.bytesize = serial.EIGHTBITS
    ser.parity = serial.PARITY_NONE
    ser.timeout = 0.5

    ser.close()

    if not ser.isOpen():
        ser.open()

    assert ser.isOpen(), 'fail to open serial'

    key = [0x01, 0x03, 0x00, 0x64, 0x00, 0x03, 0x44, 0x14]
    token = bytearray(key)

    canvas = 255 * np.ones((320, 480, 3), np.uint8)
    row, col = canvas.shape[:2]
    cv.namedWindow('Photometer_author_Steven', cv.WINDOW_NORMAL)
    cv.setWindowProperty('Photometer_author_Steven', cv.WND_PROP_TOPMOST, cv.WND_PROP_TOPMOST)

    kal = Kalman()

    while True:
        ser.write(token)
        time.sleep(0.47)
        bytes_ = ser.inWaiting()
        if not bytes_:
            break
        data = ser.read(bytes_)
        values = decode_bytes(data)
        values = kalman_filter(kal, values)

        for v in values:
            canvas.fill(255)
            cv.putText(canvas, str(round(v, 3)), (col // 6, row // 2), cv.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (0, 0, 255),
                       thickness=5)
            cv.putText(canvas, 'press ESC to quit', (col * 1 // 5, row * 3 // 4), cv.FONT_ITALIC, 1, (10, 10, 10),
                       thickness=2)
            cv.putText(canvas, '@author Steven.Zeng 2021', (col * 3 // 12, row * 28 // 30), cv.FONT_HERSHEY_COMPLEX,
                       0.5,
                       (100, 100, 100),
                       thickness=1)
            cv.imshow('Photometer_author_Steven', canvas)

        if cv.waitKey(30) == 27:
            break
    ser.close()
    cv.destroyWindow('Photometer_author_Steven')


def crc16_modbus(data):
    crc16 = crcmod.mkCrcFun(0x18005, rev=True, initCrc=0xFFFF, xorOut=0x0000)
    res = split_byte(crc16(data))
    return res[1], res[0]  # Low bit, High bit


def split_byte(d):
    return divmod(d, 0x100)


def bytearray_str(data):
    data_str = ' '.join(hex(d).upper() for d in data)
    return data_str


def decode_bytes(data):
    values = []
    head = bytearray([0x01, 0x03, 0x06])
    tail = crc16_modbus(data[:-2])

    head = bytearray_str(head)
    data_str = bytearray_str(data)
    tail = bytearray_str(tail)

    res = re.search(head + r'(.*?)' + tail, data_str, re.M | re.I)
    for res in res.groups():
        res = res.strip().split(' ')
        if len(res) == 6:
            res_ = list(reversed(res[2:6]))
            val = [int(str.encode(r), 16) for r in res_]
            val = struct.unpack('f', bytearray(val))[0]
            values.append(val)
            print(val)

    return values


def kalman_filter(kal, data):
    values = []
    for d in data:
        kal.kalman_predict(d)
        values.append(kal.statepost)
    return values


if __name__ == '__main__':
    port = input('Port=')
    pattern = re.compile(r'[1-9]')
    num = pattern.findall(port)
    if num is not None:
        port_dict = {str(i): 'COM' + str(i) for i in range(1, 11)}
        print(port_dict[num[0]])
        main(port_dict[num[0]])
