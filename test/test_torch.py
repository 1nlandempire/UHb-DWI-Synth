import numpy as np
import torch
B = 96
x = torch.randn((B, 3, 128, 128))

ret_img = x.permute(1, 0, 2, 3)
print(ret_img.shape)
ret_img = ret_img.reshape(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3])
print(ret_img.shape)
print(ret_img[-96:].shape)