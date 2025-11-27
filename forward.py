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
    parser.add_argument('-c', '--config', type=str, default='config/dMRI_sr3_128_3C_forward.json',
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
    for phase, dataset_opt in opt['datasets'].items():
        val_set = Data.create_dMRI_3C_forward(dataset_opt, phase)
        val_loader = Data.create_dataloader_forward(val_set, 96)

    idx = 0
    # 前向传播
    for _, val_data in enumerate(val_loader):

        idx += 1
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

        # affine  mask和 b0均值读取出来均为140 140 96
        mask, affine = load_nifti('/data/wtl/HCP_MGH/mgh_1011/dwi_mask.nii.gz', return_img=False)
        b0_mean_masked, b0_affine = load_nifti('/data/wtl/HCP_MGH/mgh_1011/b0_mean_masked.nii.gz', return_img=False)
        # 将b0均值裁剪并padding到128 128
        b0_mean_masked = to12812896(b0_mean_masked, mask)
        b0_mean_masked = np.expand_dims(b0_mean_masked, axis=-1)

        # 未还原
        save_nifti('./sr.nii.gz', sr_img, affine)
        save_nifti('./hr.nii.gz', hr_img, affine)
        save_nifti('./lr.nii.gz', lr_img, affine)

        # 还原
        save_nifti('./sr_b0.nii.gz', sr_img * b0_mean_masked, affine)
        save_nifti('./hr_b0.nii.gz', hr_img * b0_mean_masked, affine)
        save_nifti('./lr_b0.nii.gz', lr_img * b0_mean_masked, affine)

        # all to 128 128 1  or 128 128 3
        # sr_img = np.array(sr_img).transpose((1, 2, 0))
        # hr_img = np.array(hr_img[0, ...]).transpose((1, 2, 0))
        # lr_img = np.array(lr_img[0, ...]).transpose((1, 2, 0))
        # fake_img = np.array(fake_img[0, ...]).transpose((1, 2, 0))
        #
        # # fake img 不知道干什么用的 变为 128 128 1
        # fake_img = fake_img[..., 0:1]
        # # show
        # imshow(hr_img[:, :, 0], cmap='gray', vmin=0, vmax=1)
        # plt.show()
        # imshow(sr_img[:, :, 0], cmap='gray', vmin=0, vmax=1)
        # plt.show()
        # imshow(lr_img[:, :, 0], cmap='gray', vmin=0, vmax=1)
        # plt.show()
        # imshow(lr_img[:, :, 1], cmap='gray', vmin=0, vmax=1)
        # plt.show()
        # imshow(lr_img[:, :, 2], cmap='gray', vmin=0, vmax=1)
        # plt.show()

