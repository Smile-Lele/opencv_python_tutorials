import numpy as np


np.savez('knn_data.npz', train=np.array([[1, 2], [3, 4]]))
# Now load the data
with np.load('knn_data.npz') as data:
    print(data.files)
    train = data['train']
    print(train)