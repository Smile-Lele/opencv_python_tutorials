import cv2 as cv
from datetime import datetime
import numpy as np

if __name__ == '__main__':
    canvas = np.ones((300, 800, 3), np.uint8) * 255
    row, col = canvas.shape[:2]
    cv.namedWindow('Clock', cv.WINDOW_KEEPRATIO)
    cv.setWindowProperty('Clock', cv.WND_PROP_TOPMOST, cv.WND_PROP_TOPMOST)
    while True:
        today_ = datetime.today()
        hour = today_.hour
        minute = today_.minute
        second = today_.second
        time_str = f'{hour}:{minute}:{second}'
        cv.putText(canvas, str(time_str), (row // 3, col // 4), cv.FONT_HERSHEY_SCRIPT_COMPLEX, 4, (0, 0, 255), 6)
        cv.imshow('Clock', canvas)
        if cv.waitKey(500) & 0xFF == 27:
            break
        canvas.fill(255)

    cv.destroyWindow('Clock')
