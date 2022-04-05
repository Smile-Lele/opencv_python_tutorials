import ffmpeg
import numpy
import cv2

i = 0
while True:
    stream = ffmpeg
    stream = stream.filter('select', 'gte(n,{})'.format(i))
    stream = stream.output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
    frame, err = stream.run(capture_stdout=True)
    # frame, err = (
    #     ffmpeg
    #     .input('file_path')
    #     .filter('select', 'gte(n,{})'.format(i))
    #     .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
    #     .run(capture_stdout=True)
    # )
    if not frame:
        print('视频源链接断开', strUrl)
        time.sleep(5)
        break
        i += 1
    cv2.imshow('show', frame)
    cv2.waitKeyEx(1)
