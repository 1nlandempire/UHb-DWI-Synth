import os
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from os.path import join
import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow
from matplotlib.pyplot import imsave
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import normalized_root_mse

# z-score归一化 均值为0 方差为1 对异常值不敏感
def normalization2(data):
    mean = np.mean(data)
    std = np.std(data)

    return (data - mean) / std

# [-1,1] b1k 3k 5k 10k 的 n 分别是 700 400 300 200    b0 5000
def normalization(data, n):
    return (data * 2) / n - 1


sub_path = '/data/wtl/HCP_MGH/mgh_1001'

dwi, affine = load_nifti(join(sub_path, 'dwi.nii.gz'), return_img=False)
print(dwi.shape)

dwi0 = dwi[..., 0]
dwi14 = dwi[..., 14]

# 归一化
# dwi0 /= 5000
# dwi14 /= 5000

dwi0 = normalization(dwi0, 5000)
dwi14 = normalization(dwi14, 5000)

# 140 140
slice0 = dwi0[:, :, 48]
slice14 = dwi14[:, :, 48]


imsave('gray_image.png', slice0, cmap='gray')

imshow(slice0, cmap='gray', vmin=-1, vmax=1)
plt.show()

imshow(slice14, cmap='gray', vmin=-1, vmax=1)
plt.show()

print(slice0.shape)
print(slice14.shape)

# [-1,1]先转为[0,1]再进行计算 positive_img = (img + 1) / 2
slice0 = (slice0 + 1) / 2
slice14 = (slice14 + 1) / 2

PSNR = peak_signal_noise_ratio(slice0, slice14, data_range=1)
print(PSNR)


print('==============to 3D')

slice0 = np.expand_dims(slice0, axis=-1)
slice14 = np.expand_dims(slice14, axis=-1)
print(slice0.shape)
print(slice14.shape)
PSNR = peak_signal_noise_ratio(slice0, slice14, data_range=1)
print(PSNR)