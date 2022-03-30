import os
from mypackage.imUtils import icv
import torch
import cv2 as cv

img_arr = cv.imread('bobby.jpg', cv.IMREAD_UNCHANGED)
print(img_arr.shape)

"""
C x H x W
"""
img = torch.from_numpy(img_arr)
out = torch.transpose(img, 0, 2)

batch_size = 100
batch = torch.zeros(100, 3, 256, 256, dtype=torch.uint8)

data_dir = 'image-cats/'
filenames = [name for name in os.listdir(data_dir) if os.path.splitext(name)[-1] == '.png']
print(filenames)

for i, filename in enumerate(filenames):
    img_arr = cv.imread(os.path.join(data_dir, filename), cv.IMREAD_COLOR)
    batch[i] = torch.transpose(torch.from_numpy(img_arr), 0, 2)

# batch normalization
batch = batch.float()
channels = batch.shape[1]
for c in range(channels):
    mean = torch.mean(batch[:, c])
    std = torch.std(batch[:, c])
    batch[:, c] = (batch[:, c] - mean) / std

img1 = batch[1].transpose_(0, 2).numpy()
im_dict = dict()
im_dict['img1'] = img1
icv.implot_ex(im_dict)
