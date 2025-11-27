import os
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from os.path import join
import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow
from matplotlib.pyplot import imsave
import matplotlib.pyplot as plt

# z-score归一化 均值为0 方差为1 对异常值不敏感
def normalization2(data):
    mean = np.mean(data)
    std = np.std(data)

    return (data - mean) / std
# [-1,1] b1k 3k 5k 10k 的 n 分别是 700 400 300 200
def normalization(data, n):
    return (data * 2) / n - 1

sub_path = '/data/wtl/HCP_MGH/mgh_1001'

dwi, affine = load_nifti(join(sub_path, 'DWI_SH_b5k.nii.gz'), return_img=False)
print(dwi.shape)

# 归一化
dwi0 = normalization(dwi[..., 0], 300)
print(np.mean(dwi0))

save_nifti('normalized_DWI.nii.gz', dwi0, affine)

slice = dwi0[:, :, 48]

imsave('gray_image.png', slice, cmap='gray', vmin=-1, vmax=1)
imshow(slice, cmap='gray', vmin=-1, vmax=1)
plt.show()
