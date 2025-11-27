from dipy.io.image import load_nifti, save_nifti
import numpy as np
import os
from skimage import filters
from os.path import join
from scipy.ndimage import zoom
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt


HCP_dir = '/data/wtl/HCP_MGH'

# for sub in os.listdir(HCP_dir):
#     sub_path = join(HCP_dir, sub)
#     print(sub_path)
#
#     mask, _ = load_nifti(join(sub_path, 'dwi_mask.nii.gz'), return_img=False)
#     # print(mask.shape)
#
#     coords = np.where(mask == 1)
#     # 找出边界的上下左右前后的索引
#     xmin, xmax = np.min(coords[0]), np.max(coords[0])
#     ymin, ymax = np.min(coords[1]), np.max(coords[1])
#     zmin, zmax = np.min(coords[2]), np.max(coords[2])
#
#     # print("上边界索引：", xmin)
#     # print("下边界索引：", xmax)
#     # print("左边界索引：", ymin)
#     # print("右边界索引：", ymax)
#     # print("前边界索引：", zmin)
#     # print("后边界索引：", zmax)
#
#     new_mask = mask[xmin:xmax + 1, ymin:ymax + 1, zmin:zmax + 1]
#     print(new_mask.shape)

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
    # 原shape
    print(mask.shape)
    coords = np.where(mask == 1)
    # 找出边界的上下左右前后的索引
    xmin, xmax = np.min(coords[0]), np.max(coords[0])
    ymin, ymax = np.min(coords[1]), np.max(coords[1])
    # 裁剪
    cropped = volume[xmin:xmax + 1, ymin:ymax + 1]
    print(cropped.shape)
    # padding
    target_shape = (128, 128, 96)
    padded = padding(cropped, target_shape)
    print(padded.shape)

    imshow(volume[:, :, 48], cmap='gray')
    plt.show()

    imshow(cropped[:, :, 48], cmap='gray')
    plt.show()

    imshow(padded[:, :, 48], cmap='gray')
    plt.show()

    save_nifti('original.nii.gz', volume, affine=_)
    save_nifti('cropped.nii.gz', cropped, affine=_)
    save_nifti('padded.nii.gz', padded, affine=_)


sub_path = '/data/wtl/HCP_MGH/mgh_1001/'
mask, _ = load_nifti(join(sub_path, 'dwi_mask.nii.gz'), return_img=False)
mask = np.expand_dims(mask, axis=-1)

dwi, _ = load_nifti(join(sub_path, 'dwi.nii.gz'), return_img=False)
dwi *= mask
dwi = dwi[..., 0]
to12812896(dwi, mask)



# for sub in sorted(os.listdir(HCP_dir)):
#     print(sub)
#
#     sub_path = join(HCP_dir, sub)
#     mask, _ = load_nifti(join(sub_path, 'dwi_mask.nii.gz'), return_img=False)
#     dwi, _ = load_nifti(join(sub_path, 'dwi.nii.gz'), return_img=False)
#     dwi = dwi[..., 0]
#     to12812896(dwi, mask)

