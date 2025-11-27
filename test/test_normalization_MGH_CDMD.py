import os

import numpy as np
from dipy.io.image import load_nifti, save_nifti
from os.path import join
import matplotlib.pyplot as plt
from dipy.io import read_bvals_bvecs
# ln归一化
def clean_log(data):
    data = np.where(data < 0, 0, data)
    # nan -> 0
    data = np.where(np.isnan(data), 0, data)
    # inf -> 0
    data = np.where(np.isinf(data), 0, data)
    # 其他的异常值
    data = np.where(np.isreal(data), data, 0)
    return data

def clean(data):
    data = np.where(data > 2, 2, data)
    data = np.where(data < 0, 0, data)
    # nan -> 0
    data = np.where(np.isnan(data), 0, data)
    # 其他的异常值
    data = np.where(np.isreal(data), data, 0)
    return data

# 返回b0均值
def get_b0_mean(dwi, bval):
    # bvals 处理y数据
    bvals = np.round(bval / 100) * 100
    # b = 0 one for normalization
    indices = np.where(bvals == 0)[0]
    print(indices)
    b0s = dwi[:,:,:,indices]
    print(b0s.shape)

    b0_mean = np.mean(b0s, axis=3)

    return b0_mean

# 返回指定b值的 DWI们
def sub_sample_b(dwi, bval, specific_b):
    # bvals 处理y数据
    bvals = np.round(bval / 100) * 100
    if specific_b == 10000:
        # 9850 也归为10000
        bvals = np.round(bval / 1000) * 1000

    # b = ?
    indices = np.where(bvals == specific_b)[0]
    print(indices)
    res = dwi[:, :, :, indices]
    print(res.shape)

    return res

def normalize_dwi_HCP(sub_dir):
    print(sub_dir)
    # 140, 140, 96, n
    dwi_path = join(sub_dir, 'sub_001_dwi.nii.gz')
    mask_path = join(sub_dir, 'sub_001_dwi_brainmask.nii.gz')


    # load
    dwi, affine = load_nifti(dwi_path, return_img=False)
    mask, _ = load_nifti(mask_path, return_img=False)

    mask = np.expand_dims(mask, axis=-1)

    bval, bvec = read_bvals_bvecs(join(sub_dir, 'sub_001_dwi.bval'), join(sub_dir, 'sub_001_dwi.bvec'))

    b1000 = sub_sample_b(dwi, bval, 1000)
    b10000 = sub_sample_b(dwi, bval, 10000)
    b0mean = get_b0_mean(dwi, bval)
    b0mean = np.expand_dims(b0mean, axis=-1)

    # dwi * mask
    b1000 *= mask
    b10000 *= mask
    b0mean *= mask

    # normalization
    # b1000 /= b0mean
    # b10000 /= b0mean

    #
    # # outliers trim
    # b1000 = clean(b1000)
    # b10000 = clean(b10000)

    b1000 = np.log(b1000)
    b10000 = np.log(b10000)

    # outliers trim
    b1000 = clean_log(b1000)
    b10000 = clean_log(b10000)


    save_nifti(join(sub_dir, 'b1k_normalized.nii.gz'), b1000, affine)
    save_nifti(join(sub_dir, 'b10k_normalized.nii.gz'), b10000, affine)



# folder_path = 'D:\BaiduNetdiskDownload\processed\DWI_6dir_V2_copy'

# sub_list = sorted(os.listdir(folder_path))
# sub_list = ['654754']

normalize_dwi_HCP('/data/wtl/MGH-USC-CDMD/sub_001/dwi/')

# for file in sub_list:
#     print(join(folder_path, file))
#     if os.path.isfile(join(folder_path, file)):
#         continue
#     normalize_dwi(join(folder_path, file))

