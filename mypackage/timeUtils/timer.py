import time
from colorama import Fore, Back, Style


def clock(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(Fore.RED + f'Elapsed time: {func.__name__}() - {round(end_time - start_time, 2)}s' + Style.RESET_ALL)
        return result

    return wrapper
