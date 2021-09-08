import pandas as pd
import numpy as np


df = pd.read_excel('4K.xlsx', usecols='B:U', header=None)
data = df.iloc[2:-2, :]
mat = np.asarray(data, np.uint8)

