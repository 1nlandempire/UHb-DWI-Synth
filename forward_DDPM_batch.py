import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger_forward as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
from skimage.metrics import peak_signal_noise_ratio
from matplotlib.pyplot import imshow
from matplotlib.pyplot import imsave
import matplotlib.pyplot as plt
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from os.path import join
# 选取多个subject的多个volume 进行前向传播

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


# 前向传播 输入 (n, 3, 128, 128) 输出 (n, 1, 128, 128) b1000值
# 以后需要使用140 140 裁剪后的数据 ，不能resize


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 前向传播用
    parser.add_argument('-c', '--config', type=str, default='config/dMRI_sr3_128_3C_forward_DDPM.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # 加载模型
    print('模型加载:', opt['path']['resume_state'])
    # model
    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')

    # 数据
    # sub_list = ['mgh_1011', 'mgh_1012', 'mgh_1013', 'mgh_1014', 'mgh_1015']
    # volume_list = [
    #     [0, 38], [0], [0], [0], [0]
    # ]
    sub_list = ['mgh_1011', 'mgh_1012', 'mgh_1013', 'mgh_1014', 'mgh_1015']
    volume_list = [
        [2, 4, 6, 8, 10, 12, 14, 16],
        [2, 4, 6, 8, 10, 12, 14, 16],
        [2, 4, 6, 8, 10, 12, 14, 16],
        [2, 4, 6, 8, 10, 12, 14, 16],
        [2, 4, 6, 8, 10, 12, 14, 16]
    ]

    for phase, dataset_opt in opt['datasets'].items():
        val_set = Data.create_dMRI_3C_forward_batch(dataset_opt, phase, sub_list, volume_list)
        val_loader = Data.create_dataloader_forward(val_set, 96)

    # 前向传播 每次读取96slice 刚好是一个volume
    val_loader_iter = iter(val_loader)
    for s_idx, sub in enumerate(sub_list):
        sub_path = join('/data/wtl/HCP_MGH', sub)
        # 读取mask b0 mean等
        # affine  mask和 b0均值读取出来均为140 140 96
        mask, affine = load_nifti(join(sub_path, 'dwi_mask.nii.gz'), return_img=False)
        b0_mean_masked, b0_affine = load_nifti(join(sub_path, 'b0_mean_masked.nii.gz'), return_img=False)
        # 将b0均值裁剪并padding到128 128
        b0_mean_masked = to12812896(b0_mean_masked, mask)
        b0_mean_masked = np.expand_dims(b0_mean_masked, axis=-1)
        # 处理这个subject的每一个subject
        for volume_idx in volume_list[s_idx]:
            # save path
            home_path = '/home/wtl/workspace/Palette-Image-to-Image-Diffusion-Models/evaluation/DDPM/'
            # /home/wtl/workspace/Palette-Image-to-Image-Diffusion-Models/evaluation/SR3/mgh_1011_0
            save_path = join(home_path, f'{sub}_{volume_idx}')
            if not os.path.exists(save_path):
                # 若不存在，则直接创建目录
                os.makedirs(save_path)

            val_data = next(val_loader_iter)
            # 放入模型
            diffusion.feed_data(val_data)
            diffusion.test(continous=False)
            visuals = diffusion.get_current_visuals()

            # sr 为生成图像
            sr_img = visuals['SR']  # 1 128 128         如果dataloader设置 = 96，则torch.Size([96, 1, 128, 128])
            hr_img = visuals['HR']  # 1 1 128 128       如果dataloader设置 = 96，则torch.Size([96, 1, 128, 128])
            lr_img = visuals['LR']  # 1 3 128 128       如果dataloader设置 = 96，则torch.Size([96, 3, 128, 128])
            fake_img = visuals['INF']  # 1 3 128 128    如果dataloader设置 = 96，则torch.Size([96, 3, 128, 128])

            # 直接保存结果
            # sr hr lr 128 128 96 1; 128 128 96 1; 128 128 96 3
            sr_img = np.array(sr_img).transpose((2, 3, 0, 1))
            hr_img = np.array(hr_img).transpose((2, 3, 0, 1))
            lr_img = np.array(lr_img).transpose((2, 3, 0, 1))

            # 未还原
            save_nifti(join(save_path, 'sr.nii.gz'), sr_img, affine)
            save_nifti(join(save_path, 'hr.nii.gz'), hr_img, affine)
            save_nifti(join(save_path, 'lr.nii.gz'), lr_img, affine)

            # 还原
            save_nifti(join(save_path, 'sr_b0.nii.gz'), sr_img * b0_mean_masked, affine)
            save_nifti(join(save_path, 'hr_b0.nii.gz'), hr_img * b0_mean_masked, affine)
            save_nifti(join(save_path, 'lr_b0.nii.gz'), lr_img * b0_mean_masked, affine)


