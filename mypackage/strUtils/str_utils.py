import glob
import os


def read_multifiles(path, ext):
    files = glob.glob(os.path.join(path, f'*.{ext}'))
    assert len(files) != 0, f'fail to find {ext}'
    return files


def split_dir(file):
    dir, fname_ext = os.path.split(file)
    fname, ext = os.path.splitext(fname_ext)
    return dir, fname_ext, fname


def check_make_dir(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


def user_input_number(msg) -> float:
    PUNC_TABLE = {ord(f): ord(t) for f, t in zip(
        u'，。！？【】（）％＃＠＆１２３４５６７８９０',
        u',.!?[]()%#@&1234567890')}

    input_ = [float(k) for k in input(f'{msg} = ').translate(PUNC_TABLE).replace(' ', '').split(',')
              if k.replace('.', '').isdigit()]
    return input_
