import cv2
import torch
import imageio

dri_path = './volumetric-dicom/2-LUNG 3.0  B70f-04083/'
vol_arr = imageio.volread(dri_path, 'DICOM')
print(vol_arr.shape)

vol = torch.from_numpy(vol_arr).float()
vol = torch.transpose(vol, 0, 2)
vol = torch.unsqueeze(vol, 0)
print(vol.shape)

"""
神经网络要求将数据表示为多维数值张量，通常为32位浮点数
"""