from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
import random
import numpy as np
import h5py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from dipy.io.image import load_nifti, save_nifti
from os.path import join

# import sys
# sys.path.append("./..")

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

def clean_log(data):
    # negative
    data = np.where(data < 0, 0, data)
    # nan -> 0
    data = np.where(np.isnan(data), 0, data)
    # inf
    data = np.where(np.isinf(data), 0, data)
    # 其他的异常值
    data = np.where(np.isreal(data), data, 0)
    return data

class DMRIdataset_3C(Dataset):
    def __init__(self, dataroot=None, datatype=None, l_resolution=140, r_resolution=140, split='train', data_len=-1, need_LR=False):
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.need_LR = need_LR
        self.split = split
        self.data_len = -1

        data_path = '/data/wtl/SR_Data/'
        # 读取h5 (n, 3, 140, 140)    (n, 1, 140, 140)  已归一化大概[0 - ~1]
        self.low_b_DWI = torch.FloatTensor(np.array(h5py.File(data_path + 'low_b_DWI_3C.h5', 'r')['low_b_DWI']))
        self.high_b_DWI = torch.FloatTensor(np.array(h5py.File(data_path + 'high_b_DWI_3C.h5', 'r')['high_b_DWI']))
        # resize to 128 128
        self.low_b_DWI = F.interpolate(self.low_b_DWI, size=(128, 128), mode="bilinear", align_corners=False)
        self.high_b_DWI = F.interpolate(self.high_b_DWI, size=(128, 128), mode="bilinear", align_corners=False)
        #
        print(self.low_b_DWI.shape)
        print(self.high_b_DWI.shape)

        # n
        samples = np.arange(self.low_b_DWI.shape[0])

        # shuffle
        np.random.seed(0)
        np.random.shuffle(samples)

        self.indices = samples
        self.data_len = len(self.indices)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        low_b_DWI = self.low_b_DWI[self.indices[index], :, :, :]
        high_b_DWI = self.high_b_DWI[self.indices[index], :, :, :]

        LR = low_b_DWI
        SR = low_b_DWI
        HR = high_b_DWI

        return {'LR': low_b_DWI, 'HR': high_b_DWI, 'SR': low_b_DWI, 'Index': index}

class DMRIdataset_val_3C(Dataset):
    def __init__(self, dataroot=None, datatype=None, l_resolution=140, r_resolution=140, split='train', data_len=-1, need_LR=False):
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.need_LR = need_LR
        self.split = split
        self.data_len = 20

        data_path = '/data/wtl/SR_Data_val/'
        # 读取h5 (n, 3, 140, 140)    (n, 1, 140, 140)
        self.low_b_DWI = torch.FloatTensor(np.array(h5py.File(data_path + 'low_b_DWI_val_3C.h5', 'r')['low_b_DWI']))
        self.high_b_DWI = torch.FloatTensor(np.array(h5py.File(data_path + 'high_b_DWI_val_3C.h5', 'r')['high_b_DWI']))
        # resize to 128 128
        self.low_b_DWI = F.interpolate(self.low_b_DWI, size=(128, 128), mode="bilinear", align_corners=False)
        self.high_b_DWI = F.interpolate(self.high_b_DWI, size=(128, 128), mode="bilinear", align_corners=False)
        #
        print(self.low_b_DWI.shape)
        print(self.high_b_DWI.shape)

        # n
        samples = np.arange(self.low_b_DWI.shape[0])

        # shuffle
        np.random.seed(33)
        np.random.shuffle(samples)

        self.indices = samples[:self.data_len]


    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        low_b_DWI = self.low_b_DWI[self.indices[index], :, :, :]
        high_b_DWI = self.high_b_DWI[self.indices[index], :, :, :]

        LR = low_b_DWI
        SR = low_b_DWI
        HR = high_b_DWI

        # LR 和 SR实质上是一个东西 SR是插值后的数据，为了与HR分辨率保持一致。由于我们分辨率都一样，因此LR=SR。网络实际上以SR为输入 LR只是参考
        return {'LR': low_b_DWI, 'HR': high_b_DWI, 'SR': low_b_DWI, 'Index': index}

# crop and padding
class DMRIdataset_3C_Crop(Dataset):
    def __init__(self, dataroot=None, datatype=None, l_resolution=140, r_resolution=140, split='train', data_len=-1, need_LR=False):
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.need_LR = need_LR
        self.split = split
        self.data_len = -1

        data_path = '/data/wtl/SR_Data/'
        # 读取h5 (n, 3, 128, 128)    (n, 1, 128 128)
        self.low_b_DWI = np.array(h5py.File(data_path + 'low_b_DWI_3C_Crop_b0.h5', 'r')['low_b_DWI'])
        self.high_b_DWI = np.array(h5py.File(data_path + 'high_b_DWI_3C_Crop_b0.h5', 'r')['high_b_DWI'])

        # 额外的操作 例如取-ln()
        # self.low_b_DWI = clean_log(-np.log(self.low_b_DWI))
        # self.high_b_DWI = clean_log(-np.log(self.high_b_DWI))

        # to FloatTensor
        self.low_b_DWI = torch.FloatTensor(self.low_b_DWI)
        self.high_b_DWI = torch.FloatTensor(self.high_b_DWI)

        #
        print("DMRIdataset_3C_Crop   b0: ")
        print(self.low_b_DWI.shape)
        print(self.high_b_DWI.shape)

        # n
        samples = np.arange(self.low_b_DWI.shape[0])

        # shuffle
        np.random.seed(0)
        np.random.shuffle(samples)

        self.indices = samples
        self.data_len = len(self.indices)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        low_b_DWI = self.low_b_DWI[self.indices[index], :, :, :]
        high_b_DWI = self.high_b_DWI[self.indices[index], :, :, :]

        LR = low_b_DWI
        SR = low_b_DWI
        HR = high_b_DWI

        return {'LR': low_b_DWI, 'HR': high_b_DWI, 'SR': low_b_DWI, 'Index': index}

class DMRIdataset_val_3C_Crop(Dataset):
    def __init__(self, dataroot=None, datatype=None, l_resolution=140, r_resolution=140, split='train', data_len=-1, need_LR=False):
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.need_LR = need_LR
        self.split = split
        self.data_len = 20

        data_path = '/data/wtl/SR_Data_val/'
        # 读取h5 (n, 3, 128, 128)    (n, 1, 128, 128)
        self.low_b_DWI = np.array(h5py.File(data_path + 'low_b_DWI_val_3C_Crop_b0.h5', 'r')['low_b_DWI'])
        self.high_b_DWI = np.array(h5py.File(data_path + 'high_b_DWI_val_3C_Crop_b0.h5', 'r')['high_b_DWI'])

        # 额外的操作 例如取-ln()
        # self.low_b_DWI = clean_log(-np.log(self.low_b_DWI))
        # self.high_b_DWI = clean_log(-np.log(self.high_b_DWI))

        # to FloatTensor
        self.low_b_DWI = torch.FloatTensor(self.low_b_DWI)
        self.high_b_DWI = torch.FloatTensor(self.high_b_DWI)

        #
        print("DMRIdataset_val_3C_Crop  b0: ")
        print(self.low_b_DWI.shape)
        print(self.high_b_DWI.shape)

        # n
        samples = np.arange(self.low_b_DWI.shape[0])

        # shuffle
        np.random.seed(33)
        np.random.shuffle(samples)

        self.indices = samples[:self.data_len]


    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        low_b_DWI = self.low_b_DWI[self.indices[index], :, :, :]
        high_b_DWI = self.high_b_DWI[self.indices[index], :, :, :]

        LR = low_b_DWI
        SR = low_b_DWI
        HR = high_b_DWI

        # LR 和 SR实质上是一个东西 SR是插值后的数据，为了与HR分辨率保持一致。由于我们分辨率都一样，因此LR=SR。网络实际上以SR为输入 LR只是参考
        return {'LR': low_b_DWI, 'HR': high_b_DWI, 'SR': low_b_DWI, 'Index': index}

def subject_data_forward(sub_dir):
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

    # 选
    b1000 = b1000[:, :, :, 38:39]
    b3000 = b3000[:, :, :, 38:39]
    b5000 = b5000[:, :, :, 38:39]
    b10000 = b10000[:, :, :, 38:39]

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

def subject_data_forward_b0(sub_dir):
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

    # 选
    b1000 = b1000[:, :, :, 38:39]
    b3000 = b3000[:, :, :, 38:39]
    b5000 = b5000[:, :, :, 38:39]
    b10000 = b10000[:, :, :, 38:39]

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


# 需要传入volume list列表指定256个梯度方向里面的index  [0, 1, 38, 39]
def subject_data_forward_b0_batch(sub_dir, volume_list):
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

    # 选
    b1000 = b1000[:, :, :, volume_list]
    b3000 = b3000[:, :, :, volume_list]
    b5000 = b5000[:, :, :, volume_list]
    b10000 = b10000[:, :, :, volume_list]

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

    # low_b_DWI shape变化  140 140 96 256 to 3 140 140 96 256
    low_b_DWI = np.stack([b1000, b3000, b5000], axis=0)
    # to 3 140 140 256 96
    low_b_DWI = low_b_DWI.transpose(0, 1, 2, 4, 3)
    # to 3 140 140 96*256
    shape = (low_b_DWI.shape[0], low_b_DWI.shape[1], low_b_DWI.shape[2], low_b_DWI.shape[3] * low_b_DWI.shape[4])
    low_b_DWI = low_b_DWI.reshape(shape)
    # to 96*256 3 140 140
    low_b_DWI = low_b_DWI.transpose(3, 0, 1, 2)

    # 140 140 96 256 -> 1 140 140 96 256
    b10000 = np.expand_dims(b10000, axis=0)
    # to 1 140 140 256 96
    b10000 = b10000.transpose(0, 1, 2, 4, 3)
    # to 1 140 140 256*96
    b10000_shape = (b10000.shape[0], b10000.shape[1], b10000.shape[2], b10000.shape[3] * b10000.shape[4])
    b10000 = b10000.reshape(b10000_shape)
    # to 96*256 1 140 140
    high_b_DWI = b10000.transpose(3, 0, 1, 2)

    # low_b_DWI: (24576, 3, 140, 140)
    # high_b_DWI: (24576, 1, 140, 140)
    print('low_b_DWI:', low_b_DWI.shape)
    print('high_b_DWI:', high_b_DWI.shape)

    return low_b_DWI, high_b_DWI

# forward数据
class DMRIdataset_val_forward(Dataset):
    def __init__(self, dataroot=None, datatype=None, l_resolution=140, r_resolution=140, split='train', data_len=-1, need_LR=False):
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.need_LR = need_LR
        self.split = split

        sub_path = '/data/wtl/HCP_MGH/mgh_1001'
        # 获得(n, 3, 140, 140)    (n, 1, 140, 140)
        low_b_DWI, high_b_DWI = subject_data_forward(sub_path)
        self.low_b_DWI = torch.FloatTensor(low_b_DWI)
        self.high_b_DWI = torch.FloatTensor(high_b_DWI)
        # resize to 128 128
        self.low_b_DWI = F.interpolate(self.low_b_DWI, size=(128, 128), mode="bilinear", align_corners=False)
        self.high_b_DWI = F.interpolate(self.high_b_DWI, size=(128, 128), mode="bilinear", align_corners=False)
        #
        print(self.low_b_DWI.shape)
        print(self.high_b_DWI.shape)

        #
        self.data_len = self.low_b_DWI.shape[0]


    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        low_b_DWI = self.low_b_DWI[index, :, :, :]
        high_b_DWI = self.high_b_DWI[index, :, :, :]

        LR = low_b_DWI
        SR = low_b_DWI
        HR = high_b_DWI

        # LR 和 SR实质上是一个东西 SR是插值后的数据，为了与HR分辨率保持一致。由于我们分辨率都一样，因此LR=SR。网络实际上以SR为输入 LR只是参考
        return {'LR': low_b_DWI, 'HR': high_b_DWI, 'SR': low_b_DWI, 'Index': index}

# /b0
class DMRIdataset_val_forward_b0(Dataset):
    def __init__(self, dataroot=None, datatype=None, l_resolution=140, r_resolution=140, split='train', data_len=-1, need_LR=False):
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.need_LR = need_LR
        self.split = split

        sub_path = '/data/wtl/HCP_MGH/mgh_1011'
        # 获得(n, 3, 128, 128)    (n, 1, 128, 128)
        low_b_DWI, high_b_DWI = subject_data_forward_b0(sub_path)
        # to FloatTensor
        self.low_b_DWI = torch.FloatTensor(low_b_DWI)
        self.high_b_DWI = torch.FloatTensor(high_b_DWI)
        #
        print('DMRI forward normalized by b0')
        print(self.low_b_DWI.shape)
        print(self.high_b_DWI.shape)

        #
        self.data_len = self.low_b_DWI.shape[0]


    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        low_b_DWI = self.low_b_DWI[index, :, :, :]
        high_b_DWI = self.high_b_DWI[index, :, :, :]

        LR = low_b_DWI
        SR = low_b_DWI
        HR = high_b_DWI

        # LR 和 SR实质上是一个东西 SR是插值后的数据，为了与HR分辨率保持一致。由于我们分辨率都一样，因此LR=SR。网络实际上以SR为输入 LR只是参考
        return {'LR': low_b_DWI, 'HR': high_b_DWI, 'SR': low_b_DWI, 'Index': index}

# /b0 batch
class DMRIdataset_val_forward_b0_batch(Dataset):
    def __init__(self, sub_list, volume_list):
        print(sub_list)
        print(volume_list)
        low_b_DWI_list = []
        high_b_DWI_list = []
        for idx, sub in enumerate(sub_list):
            sub_path = join('/data/wtl/HCP_MGH/', sub)
            # 获得(n*96, 3, 128, 128)    (n*96, 1, 128, 128)
            low_b_DWI, high_b_DWI = subject_data_forward_b0_batch(sub_path, volume_list[idx])
            low_b_DWI_list.append(low_b_DWI)
            high_b_DWI_list.append(high_b_DWI)

        # list to numpy
        total_low_b_DWI = np.concatenate(low_b_DWI_list, axis=0)
        total_high_b_DWI = np.concatenate(high_b_DWI_list, axis=0)
        # to FloatTensor
        self.low_b_DWI = torch.FloatTensor(total_low_b_DWI)
        self.high_b_DWI = torch.FloatTensor(total_high_b_DWI)
        #
        print('DMRI forward normalized by b0')
        print(self.low_b_DWI.shape)
        print(self.high_b_DWI.shape)

        #
        self.data_len = self.low_b_DWI.shape[0]


    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        low_b_DWI = self.low_b_DWI[index, :, :, :]
        high_b_DWI = self.high_b_DWI[index, :, :, :]

        LR = low_b_DWI
        SR = low_b_DWI
        HR = high_b_DWI

        # LR 和 SR实质上是一个东西 SR是插值后的数据，为了与HR分辨率保持一致。由于我们分辨率都一样，因此LR=SR。网络实际上以SR为输入 LR只是参考
        return {'LR': low_b_DWI, 'HR': high_b_DWI, 'SR': low_b_DWI, 'Index': index}


if __name__ == '__main__':
    sub_list = ['mgh_1011']
    volume_list = [
        [0, 38]
    ]
    mydataset = DMRIdataset_val_forward_b0_batch(sub_list, volume_list)
    print(len(mydataset))
    for i in range(50, 60):
        slice = mydataset.__getitem__(i)
        LR = slice['LR']
        # plt.imshow(LR[0, :, :], cmap='gray', vmin=0, vmax=1)
        # plt.show()
        # plt.imshow(LR[1, :, :], cmap='gray', vmin=0, vmax=1)
        # plt.show()
        # plt.imshow(LR[2, :, :], cmap='gray', vmin=0, vmax=1)
        # plt.show()
        #
        HR = slice['HR']
        plt.imshow(HR[0, :, :], cmap='gray', vmin=0, vmax=0.5)
        plt.show()

