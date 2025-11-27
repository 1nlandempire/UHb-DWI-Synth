import os

import h5py
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from os.path import join
import matplotlib.pyplot as plt
from dipy.io import read_bvals_bvecs


# z-score归一化 [-1,1]
def normalization2(data):
    mean = np.mean(data)
    std = np.std(data)

    return (data - mean) / std

def clean(data):
    data = np.where(data > 1, 1, data)
    data = np.where(data < 0, 0, data)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data

# [-1,1] b1k 3k 5k 10k 的 n 分别是 700 400 300 200    b0 5000
def normalization(data, n):
    return (data * 2) / n - 1

# 3 channel条件
def subject_data_3C(sub_dir):
    print(sub_dir)
    # 140, 140, 96, n
    b1000_path = join(sub_dir, 'DWI_SH_b1k.nii.gz')
    b3000_path = join(sub_dir, 'DWI_SH_b3k.nii.gz')
    b5000_path = join(sub_dir, 'DWI_SH_b5k.nii.gz')

    b10000_path = join(sub_dir, 'DWI_SH_b10k.nii.gz')

    mask_path = join(sub_dir, 'dwi_mask.nii.gz')

    # load
    b1000, affine = load_nifti(b1000_path, return_img=False)
    b3000, _ = load_nifti(b3000_path, return_img=False)
    b5000, _ = load_nifti(b5000_path, return_img=False)

    b10000, _ = load_nifti(b10000_path, return_img=False)

    mask, _ = load_nifti(mask_path, return_img=False)
    mask = np.expand_dims(mask, axis=-1)
    print('mask shape:', mask.shape)
    # dwi * mask
    b1000 *= mask
    b3000 *= mask
    b5000 *= mask
    b10000 *= mask

    # 选层
    b1000 = b1000[:, :, 10:86, :]
    b3000 = b3000[:, :, 10:86, :]
    b5000 = b5000[:, :, 10:86, :]
    b10000 = b10000[:, :, 10:86, :]

    # 归一化
    b1000 = b1000 / 700
    b3000 = b3000 / 400
    b5000 = b5000 / 300
    b10000 = b10000 / 200

    # low_b_DWI shape变化  140 140 96 256 to 3 140 140 96 256
    low_b_DWI = np.stack([b1000, b3000, b5000], axis=0)
    # to 3 140 140 96*256
    shape = (low_b_DWI.shape[0], low_b_DWI.shape[1], low_b_DWI.shape[2], low_b_DWI.shape[3] * low_b_DWI.shape[4])
    low_b_DWI = low_b_DWI.reshape(shape)
    # to 96*256 3 140 140
    low_b_DWI = low_b_DWI.transpose(3, 0, 1, 2)


    # 140 140 96 256 -> 1 140 140 96*256
    b10000_shape = (1, b10000.shape[0], b10000.shape[1], b10000.shape[2] * b10000.shape[3])
    # to 96*256 1 140 140
    high_b_DWI = b10000.reshape(b10000_shape).transpose(3, 0, 1, 2)

    # low_b_DWI: (24576, 3, 140, 140)
    # high_b_DWI: (24576, 1, 140, 140)
    print('low_b_DWI:', low_b_DWI.shape)
    print('high_b_DWI:', high_b_DWI.shape)

    return low_b_DWI, high_b_DWI

# sub_dir = '/data/wtl/HCP_MGH/mgh_1001'
# low_b_DWI, high_b_DWI = subject_data(sub_dir)
# # 一个切片（特定slice，特定梯度方向）的不同b值图像
# plt.imshow(low_b_DWI[14000, 0, :, :], cmap='gray')
# plt.show()
# plt.imshow(low_b_DWI[14000, 1, :, :], cmap='gray')
# plt.show()
# plt.imshow(low_b_DWI[14000, 2, :, :], cmap='gray')
# plt.show()
#
# plt.imshow(high_b_DWI[14000, 0, :, :], cmap='gray')
# plt.show()




low_b_DWI_list = []
high_b_DWI_list = []


# path
dataset_path = '/data/wtl/HCP_MGH'
sub_list = sorted(os.listdir(dataset_path))[5:7]
print('training sub list : ', sub_list)
count = 0
for file in sub_list:
    sub_folder = join(dataset_path, file)

    low_b_DWI, high_b_DWI = subject_data_3C(sub_folder)
    low_b_DWI_list.append(low_b_DWI)
    high_b_DWI_list.append(high_b_DWI)


    count += 1

low_b_DWI_data = np.concatenate(low_b_DWI_list)
high_b_DWI_data = np.concatenate(high_b_DWI_list)

print(count)
print(low_b_DWI_data.shape)
print(high_b_DWI_data.shape)


low_b_DWI_file = h5py.File('/data/wtl/SR_Data_val/low_b_DWI_val_3C.h5', 'w')
high_b_DWI_file = h5py.File('/data/wtl/SR_Data_val/high_b_DWI_val_3C.h5', 'w')

low_b_DWI_file.create_dataset('low_b_DWI', data=low_b_DWI_data)
high_b_DWI_file.create_dataset('high_b_DWI', data=high_b_DWI_data)


