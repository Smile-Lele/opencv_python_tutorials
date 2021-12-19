import numpy as np
import pandas as pd


def write(data, file, header=None, sortby=None, ascend=None):
    """
    The function is to make it easier to save excel or csv files
    :param data: data is list of 2D or 1D
    :param file: complete file path for saved data
    :param header: optional param, the length of header can be ambiguous
    :param sortby: optional param, default None
    :param ascend: optional param, default None
    :return: None
    """

    # init dataframe
    df = pd.DataFrame(data)

    # complete header and set header for dataframe
    if header:
        noheader_cols = df.shape[1] - len(header)
        complete_header = ['P' + str(i) for i in range(1, noheader_cols + 1)]
        header += complete_header
        df.columns = header

    # sort dataframe
    if sortby:
        df = df.sort_values(by=sortby, ascending=ascend)

    # select the type of saved file
    if '.csv' in file:
        df.to_csv(file, index=False, encoding='utf8')
    elif '.xlsx' in file:
        with pd.ExcelWriter(file) as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False, encoding='utf8')


def read(file, userows: str = None, usecols: str = None):
    df = pd.DataFrame()

    if '.csv' in file:
        df = pd.read_csv(file)
    elif '.xlsx' in file:
        df = pd.read_excel(file)

    data = np.asarray(df)

    if userows:
        start, end = userows.split(':')
        if start != ' ':
            data = data[int(start) - 2:, :]
        if end != ' ':
            data = data[:int(end) - 2, :]
    if usecols:
        start, end = usecols.split(':')
        if start != ' ':
            data = data[:, int(ord(start) - ord('A')):]
        if end != ' ':
            data = data[:, :int(ord(end) - ord('A'))]
    print(f'in:{df.shape} -> out:{data.shape}')
    return data


def test():
    df = read('4k.xlsx', userows='3:12', usecols='B:U')
    print(np.array(df))


if __name__ == '__main__':
    test()
