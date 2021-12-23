'''
Compute MMD distance
domain adaptation 的 loss
参考：
    知乎专栏：https://zhuanlan.zhihu.com/p/53359505
    https://blog.csdn.net/a529975125/article/details/81176029
    论文：
    Integrating structured biological data by Kernel Maximum Mean Discrepancy
    迁移学习简明手册：
        https://zhuanlan.zhihu.com/p/35352154
    awesome domain adaptation:
        https://github.com/zhaoxin94/awesome-domain-adaptation
'''

import torch
import torch.nn as nn


class MMD_loss(nn.Module):
    '''
    最大均值差异（Maximum Mean Discrepancy, MMD）
    '''
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        '''
        高斯核: exp{\frac{(x_i-x_j)^2}{\sigma}}
        使用多个不同的高斯核，作为映射
        '''
        n_samples = int(source.size()[0]) + int(target.size()[0]) # 两个batch size相加
        # 此处将循环改成了向量操作，得到 L2 距离。
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # 得到一个矩阵 size(0) * size(0)
        L2_distance = ((total0-total1)**2).sum(2)
        # 标准差
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        # 得到多个核函数矩阵
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        # 5个核矩阵的和，得到一个核矩阵
        return sum(kernel_val)

    def linear_mmd(self, f_of_X, f_of_Y):
        '''
        线性核函数，
        f_of_X: source data
        f_of_Y: target data
        '''
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        # delta 是一个向量，距离越小越好
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        '''
        计算 loss
        '''
        if self.kernel_type == 'linear':
            return self.linear_mmd(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                # 核方法版本的最大均值差异
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            torch.cuda.empty_cache()
            return loss
