import os

import h5py
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from os.path import join
import matplotlib.pyplot as plt
from dipy.io import read_bvals_bvecs

# crop b0归一化
def padding(data, target_shape):


    # 计算在每个维度上的填充量
    x = (target_shape[0] - data.shape[0]) // 2
    y = (target_shape[1] - data.shape[1]) // 2
    z = (target_shape[2] - data.shape[2]) // 2


    pad_width = [x,x,y,y,z,z]
    if data.shape[0] % 2 != 0:
        pad_width[0] += 1
    if data.shape[1] % 2 != 0:
        pad_width[2] += 1
    if data.shape[2] % 2 != 0:
        pad_width[4] += 1

    # 进行填充，使用常数值0填充
    padded_data = np.pad(data,
                         ((pad_width[0], pad_width[1]), (pad_width[2], pad_width[3]), (pad_width[4], pad_width[5])),
                         mode='constant', constant_values=0)


    return padded_data

# crop and padding 无失真 需要根据mask进行crop 一个subject用一个mask
def to12812896(volume, mask):
    mask = mask[..., 0]
    # 原shape
    # print(mask.shape)
    coords = np.where(mask == 1)
    # 找出边界的上下左右前后的索引
    xmin, xmax = np.min(coords[0]), np.max(coords[0])
    ymin, ymax = np.min(coords[1]), np.max(coords[1])
    # 裁剪
    cropped = volume[xmin:xmax + 1, ymin:ymax + 1]
    # print(cropped.shape)
    # padding
    target_shape = (128, 128, 96)
    padded = padding(cropped, target_shape)
    # print(padded.shape)

    return padded

def clean(data):
    data = np.where(data > 2, 2, data)
    data = np.where(data < 0, 0, data)
    # nan -> 0
    data = np.where(np.isnan(data), 0, data)
    # 其他的异常值
    data = np.where(np.isreal(data), data, 0)
    return data

# 3 channel条件 crop
def subject_data_3C_crop_b0(sub_dir):
    print(sub_dir)
    # 140, 140, 96, n
    b1000_path = join(sub_dir, 'DWI_SH_b1k.nii.gz')
    b3000_path = join(sub_dir, 'DWI_SH_b3k.nii.gz')
    b5000_path = join(sub_dir, 'DWI_SH_b5k.nii.gz')
    b10000_path = join(sub_dir, 'DWI_SH_b10k.nii.gz')
    mask_path = join(sub_dir, 'dwi_mask.nii.gz')
    b0mean_path = join(sub_dir, 'b0_mean_masked.nii.gz')

    # load
    b1000, affine = load_nifti(b1000_path, return_img=False)
    b3000, _ = load_nifti(b3000_path, return_img=False)
    b5000, _ = load_nifti(b5000_path, return_img=False)
    b10000, _ = load_nifti(b10000_path, return_img=False)
    mask, _ = load_nifti(mask_path, return_img=False)
    b0mean, _ = load_nifti(b0mean_path, return_img=False)

    mask = np.expand_dims(mask, axis=-1)
    b0mean = np.expand_dims(b0mean, axis=-1)
    # print('mask shape:', mask.shape)

    # 选取前128个volume
    b1000 = b1000[..., :128]
    b3000 = b3000[..., :128]
    b5000 = b5000[..., :128]
    b10000 = b10000[..., :128]

    # dwi * mask
    b1000 *= mask
    b3000 *= mask
    b5000 *= mask
    b10000 *= mask

    # normalization
    b1000 /= b0mean
    b3000 /= b0mean
    b5000 /= b0mean
    b10000 /= b0mean

    # outliers trim
    b1000 = clean(b1000)
    b3000 = clean(b3000)
    b5000 = clean(b5000)
    b10000 = clean(b10000)

    # crop and padding
    b1000_crop = np.zeros((128, 128, 96, b1000.shape[3]))
    b3000_crop = np.zeros((128, 128, 96, b3000.shape[3]))
    b5000_crop = np.zeros((128, 128, 96, b5000.shape[3]))
    b10000_crop = np.zeros((128, 128, 96, b10000.shape[3]))
    for v in range(b1000.shape[3]):
        b1000_crop[..., v] = to12812896(b1000[..., v], mask)
        b3000_crop[..., v] = to12812896(b3000[..., v], mask)
        b5000_crop[..., v] = to12812896(b5000[..., v], mask)
        b10000_crop[..., v] = to12812896(b10000[..., v], mask)
    #
    b1000 = b1000_crop
    b3000 = b3000_crop
    b5000 = b5000_crop
    b10000 = b10000_crop

    # 选层
    b1000 = b1000[:, :, 11:86, :]
    b3000 = b3000[:, :, 11:86, :]
    b5000 = b5000[:, :, 11:86, :]
    b10000 = b10000[:, :, 11:86, :]


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

low_b_DWI_list = []
high_b_DWI_list = []


# path
dataset_path = '/data/wtl/HCP_MGH'
sub_list = sorted(os.listdir(dataset_path))[10:12]
print('training sub list : ', sub_list)
count = 0
for file in sub_list:
    sub_folder = join(dataset_path, file)

    low_b_DWI, high_b_DWI = subject_data_3C_crop_b0(sub_folder)
    low_b_DWI_list.append(low_b_DWI)
    high_b_DWI_list.append(high_b_DWI)


    count += 1

low_b_DWI_data = np.concatenate(low_b_DWI_list)
high_b_DWI_data = np.concatenate(high_b_DWI_list)

print(count)
print(low_b_DWI_data.shape)
print(high_b_DWI_data.shape)


low_b_DWI_file = h5py.File('/data/wtl/SR_Data_val/low_b_DWI_val_3C_Crop_b0.h5', 'w')
high_b_DWI_file = h5py.File('/data/wtl/SR_Data_val/high_b_DWI_val_3C_Crop_b0.h5', 'w')

low_b_DWI_file.create_dataset('low_b_DWI', data=low_b_DWI_data)
high_b_DWI_file.create_dataset('high_b_DWI', data=high_b_DWI_data)


