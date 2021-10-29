import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.functional as f
from torch.autograd import Variable


data = [[1,2],[3,4]]
np_d = np.array(data)
tensor_d = torch.from_numpy(np_d)
print(tensor_d)