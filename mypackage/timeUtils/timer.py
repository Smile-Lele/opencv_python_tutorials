import time


def clock(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f'Elapsed time: {func.__name__}() - {round(end_time - start_time, 2)}s')
        return result

    return wrapper
