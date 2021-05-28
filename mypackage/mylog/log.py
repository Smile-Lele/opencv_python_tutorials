# coding: utf-8

import logging


def init_logger(path='back.log'):
    logger = logging.getLogger('mylogger')
    logger.setLevel('DEBUG')

    # create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel('INFO')

    SIMPLE_FORMAT = '%(process)d [L:%(lineno)d] %(levelname)s %(message)s'
    console_formatter = logging.Formatter(SIMPLE_FORMAT)
    console_handler.setFormatter(console_formatter)

    # create file handler
    file_handler = logging.FileHandler(path)
    file_handler.setLevel('DEBUG')

    BASIC_FORMAT = '%(asctime)s %(filename)s %(process)d:%(threadName)s-%(thread)d [L:%(lineno)d] %(levelname)s %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    file_formatter = logging.Formatter(BASIC_FORMAT)
    file_handler.setFormatter(file_formatter)

    # add handler
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = init_logger()

if __name__ == '__main__':
    logger.info('this is msg')
    logger.debug('msg')
    logger.error('this is msg')
