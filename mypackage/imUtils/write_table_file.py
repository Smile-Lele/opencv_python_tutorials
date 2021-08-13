import pandas as pd


def write_file(data, file, header=None, sortby=None, ascend=None):
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
    noheader_cols = df.shape[1] - len(header)
    complete_header = ['P' + str(i) for i in range(1, noheader_cols + 1)]
    header += complete_header
    df.columns = header

    # sort dataframe
    df = df.sort_values(by=sortby, ascending=ascend)

    # select the type of saved file
    if '.csv' in file:
        df.to_csv(file, index=False, encoding='utf8')
    elif '.xlsx' in file:
        with pd.ExcelWriter(file) as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False, encoding='utf8')

