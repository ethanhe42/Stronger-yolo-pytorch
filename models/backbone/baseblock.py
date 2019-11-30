import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torch


class conv_bn(nn.Module):
    def __init__(self, inp, oup, kernel, stride, padding, activate='relu6'):
        super(conv_bn, self).__init__()
        if activate == 'relu6':
            self.convbn = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(inp, oup, kernel, stride, padding, bias=False)),
                ('bn', nn.BatchNorm2d(oup)),
                ('relu', nn.ReLU6(inplace=True))
            ]))
        elif activate == 'leaky':
            self.convbn = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(inp, oup, kernel, stride, padding, bias=False)),
                ('bn', nn.BatchNorm2d(oup)),
                ('relu', nn.LeakyReLU(0.1))
            ]))
        else:
            raise AttributeError("activate type not supported")

    def forward(self, input):
        return self.convbn(input)


class ASFF(nn.Module):
    def __init__(self, level, activate, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [512, 256, 128]
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = conv_bn(256, self.inter_dim, kernel=3, stride=2, padding=1, activate=activate)
            self.stride_level_2 = conv_bn(128, self.inter_dim, kernel=3, stride=2, padding=1, activate=activate)
            self.expand = conv_bn(self.inter_dim, 512, kernel=3, stride=1, padding=1, activate=activate)
        elif level == 1:
            self.compress_level_0 = conv_bn(512, self.inter_dim, kernel=1, stride=1, padding=0, activate=activate)
            self.stride_level_2 = conv_bn(128, self.inter_dim, kernel=3, stride=2, padding=1, activate=activate)
            self.expand = conv_bn(self.inter_dim, 256, kernel=3, stride=1, padding=1, activate=activate)
        elif level == 2:
            self.compress_level_0 = conv_bn(512, self.inter_dim, kernel=1, stride=1, padding=0, activate=activate)
            self.compress_level_1= conv_bn(256,self.inter_dim,kernel=1,stride=1,padding=0,activate=activate)
            self.expand = conv_bn(self.inter_dim, 128, kernel=3, stride=1, padding=1, activate=activate)

        compress_c = 8 if rfb else 16  # when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = conv_bn(self.inter_dim, compress_c, 1, 1, 0, activate=activate)
        self.weight_level_1 = conv_bn(self.inter_dim, compress_c, 1, 1, 0, activate=activate)
        self.weight_level_2 = conv_bn(self.inter_dim, compress_c, 1, 1, 0, activate=activate)

        self.weight_levels = conv_bias(compress_c * 3, 3, kernel=1, stride=1, padding=0)
        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


class conv_bias(nn.Module):
    def __init__(self, inp, oup, kernel, stride, padding):
        super(conv_bias, self).__init__()
        self.conv = nn.Conv2d(inp, oup, kernel, stride, padding, bias=True)

    def forward(self, input):
        return self.conv(input)


class sepconv_bn(nn.Module):
    def __init__(self, inp, oup, kernel, stride, padding, seprelu):
        super(sepconv_bn, self).__init__()
        if seprelu:
            self.sepconv_bn = nn.Sequential(OrderedDict([
                ('sepconv', nn.Conv2d(inp, inp, kernel, stride, padding, groups=inp, bias=False)),
                ('sepbn', nn.BatchNorm2d(inp)),
                ('seprelu', nn.ReLU6(inplace=True)),
                ('pointconv', nn.Conv2d(inp, oup, 1, 1, 0, bias=False)),
                ('pointbn', nn.BatchNorm2d(oup)),
                ('pointrelu', nn.ReLU6(inplace=True)),
            ]))
        else:
            self.sepconv_bn = nn.Sequential(OrderedDict([
                ('sepconv', nn.Conv2d(inp, inp, kernel, stride, padding, groups=inp, bias=False)),
                ('sepbn', nn.BatchNorm2d(inp)),
                ('pointconv', nn.Conv2d(inp, oup, 1, 1, 0, bias=False)),
                ('pointbn', nn.BatchNorm2d(oup)),
                ('pointrelu', nn.ReLU6(inplace=True)),
            ]))

    def forward(self, input):
        return self.sepconv_bn(input)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(OrderedDict([
                ('dw_conv', nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)),
                ('dw_bn', nn.BatchNorm2d(hidden_dim)),
                ('dw_relu', nn.ReLU6(inplace=True)),
                ('project_conv', nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)),
                ('project_bn', nn.BatchNorm2d(oup))
            ]))
        else:
            self.conv = nn.Sequential(OrderedDict(
                [
                    ('expand_conv', nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)),
                    ('expand_bn', nn.BatchNorm2d(hidden_dim)),
                    ('expand_relu', nn.ReLU6(inplace=True)),
                    ('dw_conv', nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)),
                    ('dw_bn', nn.BatchNorm2d(hidden_dim)),
                    ('dw_relu', nn.ReLU6(inplace=True)),
                    ('project_conv', nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)),
                    ('project_bn', nn.BatchNorm2d(oup))
                ]
            )
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class DarknetBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(DarknetBlock, self).__init__()
        self.darkblock = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(inplanes, planes[0], kernel_size=1,
                                stride=1, padding=0, bias=False)),
            ('bn1', nn.BatchNorm2d(planes[0])),
            ('relu1', nn.LeakyReLU(0.1)),
            ('project_conv', nn.Conv2d(planes[0], planes[1], kernel_size=3,
                                       stride=1, padding=1, bias=False)),
            ('project_bn', nn.BatchNorm2d(planes[1])),
            ('project_relu', nn.LeakyReLU(0.1)),
        ]))

    def forward(self, x):
        out = self.darkblock(x)
        out += x
        return out
if __name__ == '__main__':
    model=ASFF(1,activate='leaky')
    l1=torch.ones(1,512,10,10)
    l2=torch.ones(1,256,20,20)
    l3=torch.ones(1,128,40,40)
    out=model(l1,l2,l3)
    print(out.shape)