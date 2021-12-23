# -*- coding: UTF-8 -*-
""" inceptionv4 in pytorch
[1] Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
    Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
    https://arxiv.org/abs/1602.07261
"""

import torch
import torch.nn as nn


class BasicConv2d(nn.Module):
    '''
    卷积 -> batchnorm -> 激活函数
    '''
    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Inception_Stem(nn.Module):
    '''
    stem 模块
    输入大小: 3*256*256
    输出大小: 384*30*30
    '''
    def __init__(self, input_channels):
        super().__init__()
        # 图片原始大小 256
        self.conv1 = nn.Sequential(
            BasicConv2d(input_channels, 32, stride=2, kernel_size=3), # 128
            BasicConv2d(32, 32, kernel_size=3), # 126
            BasicConv2d(32, 64, kernel_size=3, padding=1) # 126
        )

        self.branch3x3_conv = BasicConv2d(64, 96, stride=2, kernel_size=3, padding=1) # 62
        self.branch3x3_pool = nn.MaxPool2d(3, stride=2, padding=1) # 62

        self.branch7x7a = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1), # 60
            BasicConv2d(64, 64, kernel_size=(7, 1), padding=(3, 0)), # 60
            BasicConv2d(64, 64, kernel_size=(1, 7), padding=(0, 3)), # 60
            BasicConv2d(64, 96, kernel_size=3, padding=1)
        )

        self.branch7x7b = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1), # 60
            BasicConv2d(64, 96, kernel_size=3, padding=1) # 60
        )

        self.branchpoola = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 30
        self.branchpoolb = BasicConv2d(192, 192, kernel_size=3, stride=2, padding=1) # 30

    def forward(self, x):

        x = self.conv1(x)

        x = [
            self.branch3x3_conv(x),
            self.branch3x3_pool(x)
        ]
        x = torch.cat(x, 1)

        x = [
            self.branch7x7a(x),
            self.branch7x7b(x)
        ]
        x = torch.cat(x, 1)

        x = [
            self.branchpoola(x),
            self.branchpoolb(x)
        ]

        x = torch.cat(x, 1)

        return x


class InceptionA(nn.Module):
    '''
    inception-A 模块
    输入：384*30*30
    输出：384*30*30
    '''
    def __init__(self, input_channels):
        super().__init__()

        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, 64, kernel_size=1), # 30
            BasicConv2d(64, 96, kernel_size=3, padding=1), # 30
            BasicConv2d(96, 96, kernel_size=3, padding=1) # 30
        )

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 64, kernel_size=1), # 30
            BasicConv2d(64, 96, kernel_size=3, padding=1) # 30
        )

        self.branch1x1 = BasicConv2d(input_channels, 96, kernel_size=1) # 30

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1), # 30
            BasicConv2d(input_channels, 96, kernel_size=1) # 30
        )

    def forward(self, x):

        x = [
            self.branch3x3stack(x),
            self.branch3x3(x),
            self.branch1x1(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)

class ReductionA(nn.Module):

    '''
    Reduction A 模块
    输入：384*30*30
    输出：(input_channels + n + m) * 15 * 15
    '''
    def __init__(self, input_channels, k, l, m, n):

        super().__init__()
        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, k, kernel_size=1), # 30
            BasicConv2d(k, l, kernel_size=3, padding=1), # 30
            BasicConv2d(l, m, kernel_size=3, stride=2) # 15
        )

        self.branch3x3 = BasicConv2d(input_channels, n, kernel_size=3, stride=2) # 15
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2) # 15
        self.output_channels = input_channels + n + m

    def forward(self, x):

        x = [
            self.branch3x3stack(x),
            self.branch3x3(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)


class InceptionB(nn.Module):
    '''
    Inception-B 模块
    输入: (input_channels + n + m) * 15 * 15
    输出: 1024 * 15 * 15
    '''
    def __init__(self, input_channels):
        super().__init__()

        self.branch7x7stack = nn.Sequential(
            BasicConv2d(input_channels, 192, kernel_size=1), # 15
            BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3)), # 15
            BasicConv2d(192, 224, kernel_size=(7, 1), padding=(3, 0)), # 15
            BasicConv2d(224, 224, kernel_size=(1, 7), padding=(0, 3)), # 15
            BasicConv2d(224, 256, kernel_size=(7, 1), padding=(3, 0)) # 15
        )

        self.branch7x7 = nn.Sequential(
            BasicConv2d(input_channels, 192, kernel_size=1), # 15
            BasicConv2d(192, 224, kernel_size=(1, 7), padding=(0, 3)), # 15
            BasicConv2d(224, 256, kernel_size=(7, 1), padding=(3, 0)) # 15
        )

        self.branch1x1 = BasicConv2d(input_channels, 384, kernel_size=1) # 15

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1), # 15
            BasicConv2d(input_channels, 128, kernel_size=1) # 15
        )

    def forward(self, x):
        x = [
            self.branch1x1(x),
            self.branch7x7(x),
            self.branch7x7stack(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)

class ReductionB(nn.Module):

    '''
    输入: 1024 * 15 * 15
    输出: 1536 * 7 * 7
    '''
    def __init__(self, input_channels):

        super().__init__()
        self.branch7x7 = nn.Sequential(
            BasicConv2d(input_channels, 256, kernel_size=1), # 15
            BasicConv2d(256, 256, kernel_size=(1, 7), padding=(0, 3)), # 15
            BasicConv2d(256, 320, kernel_size=(7, 1), padding=(3, 0)), # 15
            BasicConv2d(320, 320, kernel_size=3, stride=2, padding=1) # 7
        )

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 192, kernel_size=1), # 15
            BasicConv2d(192, 192, kernel_size=3, stride=2, padding=1) # 7
        )

        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 7

    def forward(self, x):

        x = [
            self.branch3x3(x),
            self.branch7x7(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)

class InceptionC(nn.Module):

    def __init__(self, input_channels):
        '''
        输入: 1536 * 7 * 7
        输出: 1536 * 7 * 7
        '''
        super().__init__()

        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, 384, kernel_size=1), # 7
            BasicConv2d(384, 448, kernel_size=(1, 3), padding=(0, 1)), # 7
            BasicConv2d(448, 512, kernel_size=(3, 1), padding=(1, 0)), # 7
        )
        self.branch3x3stacka = BasicConv2d(512, 256, kernel_size=(1, 3), padding=(0, 1)) # 7
        self.branch3x3stackb = BasicConv2d(512, 256, kernel_size=(3, 1), padding=(1, 0)) # 7
    
        self.branch3x3 = BasicConv2d(input_channels, 384, kernel_size=1) # 7
        self.branch3x3a = BasicConv2d(384, 256, kernel_size=(3, 1), padding=(1, 0)) # 7
        self.branch3x3b = BasicConv2d(384, 256, kernel_size=(1, 3), padding=(0, 1)) # 7

        self.branch1x1 = BasicConv2d(input_channels, 256, kernel_size=1) # 7

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1), # 7
            BasicConv2d(input_channels, 256, kernel_size=1) # 7
        )

    def forward(self, x):
        branch3x3stack_output = self.branch3x3stack(x)
        branch3x3stack_output = [
            self.branch3x3stacka(branch3x3stack_output),
            self.branch3x3stackb(branch3x3stack_output)
        ]
        branch3x3stack_output = torch.cat(branch3x3stack_output, 1)

        branch3x3_output = self.branch3x3(x)
        branch3x3_output = [
            self.branch3x3a(branch3x3_output),
            self.branch3x3b(branch3x3_output)
        ]
        branch3x3_output = torch.cat(branch3x3_output, 1)

        branch1x1_output = self.branch1x1(x)

        branchpool = self.branchpool(x)

        output = [
            branch1x1_output,
            branch3x3_output,
            branch3x3stack_output,
            branchpool
        ]

        return torch.cat(output, 1)
        
class InceptionV4(nn.Module):

    def __init__(self, A, B, C, k=192, l=224, m=256, n=384, class_nums=38):

        super().__init__()
        self.stem = Inception_Stem(3)
        self.inception_a = self._generate_inception_module(384, 384, A, InceptionA)
        self.reduction_a = ReductionA(384, k, l, m, n)
        output_channels = self.reduction_a.output_channels
        self.inception_b = self._generate_inception_module(output_channels, 1024, B, InceptionB)
        self.reduction_b = ReductionB(1024)
        self.inception_c = self._generate_inception_module(1536, 1536, C, InceptionC)
        self.avgpool = nn.AvgPool2d(7)

        #"""Dropout (keep 0.8)"""
        self.dropout = nn.Dropout2d(1 - 0.8)
        self.linear = nn.Linear(1536, class_nums)

        self.features_map = None

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(-1, 1536)

        self.features_map = x.detach()

        x = self.linear(x)

        return x

    @staticmethod
    def _generate_inception_module(input_channels, output_channels, block_num, block):
        '''
        连续添加模块
        '''
        layers = nn.Sequential()
        for l in range(block_num):
            layers.add_module("{}_{}".format(block.__name__, l), block(input_channels))
            input_channels = output_channels
        return layers

def inceptionv4():
    return InceptionV4(4, 7, 3)
