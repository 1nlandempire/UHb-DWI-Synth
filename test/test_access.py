import os
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from os.path import join
import numpy as np
from PIL import Image

# 数据
sub_list = ['mgh_1011', 'mgh_1012', 'mgh_1013', 'mgh_1014', 'mgh_1015']
volume_list = [
    [0], [0], [0], [0], [0]
]



for s_idx, sub in enumerate(sub_list):

    # 处理这个subject的每一个subject
    for volume_idx in volume_list[s_idx]:
        # save path
        home_path = '/home/wtl/workspace/Palette-Image-to-Image-Diffusion-Models/evaluation/SR3/'
        save_path = join(home_path, f'{sub}_{volume_idx}')
        print(save_path)

