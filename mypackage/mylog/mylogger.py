import logging
import os
import time


class ILogger(logging.Logger):
    def __init__(self, path=None):
        super().__init__(name='Log')
        self.setLevel(logging.DEBUG)

        short_fmt = '%(filename)s %(processName)s:%(threadName)s [L:%(lineno)d] %(levelname)s %(message)s'
        short_fmtter = logging.Formatter(short_fmt)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(short_fmtter)
        self.addHandler(stream_handler)

        if path:
            uuid_str = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
            file = '%s.log' % uuid_str
            file_handler = logging.FileHandler(os.path.join(path, file))
            file_handler.setLevel(logging.WARNING)
            long_fmt = '%(asctime)s ' + short_fmt
            long_fmtter = logging.Formatter(long_fmt)
            file_handler.setFormatter(long_fmtter)
            self.addHandler(file_handler)


if __name__ == '__main__':
    logger = ILogger(os.getcwd())
    logger.debug("world")
    logger.warning('Renee')
    logger.debug('Renee')
    logger.critical('Renee')
