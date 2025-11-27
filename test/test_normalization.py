import os

import numpy as np
from dipy.io.image import load_nifti, save_nifti
from os.path import join
import matplotlib.pyplot as plt
from dipy.io import read_bvals_bvecs

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

# 返回b0均值
def sub_sample(dwi, bval):
    # bvals 处理y数据
    bvals = np.round(bval / 100) * 100
    # b = 0 one for normalization
    indices = np.where(bvals == 0)[0]

    b0s = dwi[:,:,:,indices]
    print(b0s.shape)

    b0_mean = np.mean(b0s, axis=3)

    return b0_mean


def normalize_dwi(sub_dir):
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

    save_nifti(join(sub_dir, 'b1k_normalized.nii.gz'), b1000, affine)
    save_nifti(join(sub_dir, 'b3k_normalized.nii.gz'), b3000, affine)
    save_nifti(join(sub_dir, 'b5k_normalized.nii.gz'), b5000, affine)
    save_nifti(join(sub_dir, 'b10k_normalized.nii.gz'), b10000, affine)

# folder_path = 'D:\BaiduNetdiskDownload\processed\DWI_6dir_V2_copy'

# sub_list = sorted(os.listdir(folder_path))
# sub_list = ['654754']

normalize_dwi('/data/wtl/HCP_MGH/mgh_1001')

# for file in sub_list:
#     print(join(folder_path, file))
#     if os.path.isfile(join(folder_path, file)):
#         continue
#     normalize_dwi(join(folder_path, file))

