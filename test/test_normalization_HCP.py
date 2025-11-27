import os

import numpy as np
from dipy.io.image import load_nifti, save_nifti
from os.path import join
import matplotlib.pyplot as plt
from dipy.io import read_bvals_bvecs


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

# 返回指定b值的 DWI们
def sub_sample_b(dwi, bval, specific_b):
    # bvals 处理y数据
    bvals = np.round(bval / 100) * 100
    # b = 1000
    indices = np.where(bvals == specific_b)[0]
    res = dwi[:, :, :, indices]
    print(res.shape)

    return res

def normalize_dwi_HCP(sub_dir):
    print(sub_dir)
    # 140, 140, 96, n
    dwi_path = join(sub_dir, 'data.nii.gz')
    mask_path = join(sub_dir, 'nodif_brain_mask.nii.gz')
    b0mean_path = join(sub_dir, 'b0_mean_masked.nii.gz')

    # load
    dwi, affine = load_nifti(dwi_path, return_img=False)
    mask, _ = load_nifti(mask_path, return_img=False)
    b0mean, _ = load_nifti(b0mean_path, return_img=False)

    mask = np.expand_dims(mask, axis=-1)
    b0mean = np.expand_dims(b0mean, axis=-1)
    bval, bvec = read_bvals_bvecs(join(sub_dir, 'bvals'), join(sub_dir, 'bvecs'))

    b1000 = sub_sample_b(dwi, bval, 1000)
    b2000 = sub_sample_b(dwi, bval, 2000)
    b3000 = sub_sample_b(dwi, bval, 3000)

    # dwi * mask
    b1000 *= mask
    b2000 *= mask
    b3000 *= mask

    # normalization
    b1000 /= b0mean
    b2000 /= b0mean
    b3000 /= b0mean

    # outliers trim
    b1000 = clean(b1000)
    b2000 = clean(b2000)
    b3000 = clean(b3000)

    save_nifti(join(sub_dir, 'b1k_normalized.nii.gz'), b1000, affine)
    save_nifti(join(sub_dir, 'b2k_normalized.nii.gz'), b2000, affine)
    save_nifti(join(sub_dir, 'b3k_normalized.nii.gz'), b3000, affine)


# folder_path = 'D:\BaiduNetdiskDownload\processed\DWI_6dir_V2_copy'

# sub_list = sorted(os.listdir(folder_path))
# sub_list = ['654754']

normalize_dwi_HCP('/data/wtl/HCP_microstructure/100610/')

# for file in sub_list:
#     print(join(folder_path, file))
#     if os.path.isfile(join(folder_path, file)):
#         continue
#     normalize_dwi(join(folder_path, file))

