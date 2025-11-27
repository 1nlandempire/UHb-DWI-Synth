
import torch
import torch.nn as nn
class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv1d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm1d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class StarBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=3):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 3, 1, (3 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 3, 1, (3 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.scale * x
        return x


starblock = StarBlock(4)
x = torch.randn((2, 4, 128, 128))
y = starblock(x)
print(y.shape)