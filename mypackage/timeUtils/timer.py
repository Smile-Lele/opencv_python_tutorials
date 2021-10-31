# 可监控程序运行时间
import time


def clock(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("共耗时: %s秒" % round(end_time - start_time, 2))
        return result

    return wrapper
