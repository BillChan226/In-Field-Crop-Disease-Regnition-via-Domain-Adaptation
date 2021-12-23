'''
vgg 模块
参考：https://github.com/jiweibo/ImageNet
'''
import math
import torch.nn as nn

# 图片大小依次为：256->128->64->32->16->8
cfg = [32, 'M', 64, 'M', 128, 'M', 256, 'M', 256, 256, 'M']


class VGG(nn.Module):
    '''
    vgg
    '''
    def __init__(self, features, num_classes=38, init_weights=True):
        super(VGG, self).__init__()
        # 卷积提取特征
        self.features = features
        # 最后的分类器
        self.classifier = nn.Sequential(nn.Linear(256 * 8 * 8, 512),
                                        nn.ReLU(inplace=True), nn.Dropout(),
                                        nn.Linear(512, 128),
                                        nn.ReLU(inplace=True), nn.Dropout(),
                                        nn.Linear(128, num_classes))
        # 权重初始化, 把上面定义的 feature 和 classifier 都初始化
        if init_weights:
            self._initialize_weights()
        # 需要的模型的特征
        self.features_map = None

    def forward(self, x):
        x = self.features(x)
        # x.size(0) 是一个batch大小
        x = x.view(x.size(0), -1)
        self.features_map = x.detach() # x 大小 batch_size * 256 * 8 * 8
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
                    # nn.init.constant_(m.bias.data, val=0.0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # nn.init.constant_(m.weight.data, val=1.0)
                # nn.init.constant_(m.bias.data, val=0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                # nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
                # nn.init.constant_(m.bias.data, val=0.0)


def make_layers(config, batch_norm=False):
    '''
    构建特征层
    config 列表: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M']
    '''
    layers = []
    in_channels = 3
    for v in config:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # in_channels: 上一层卷积核个数, v: 卷积核个数, 3*3的卷积核, padding=1 确保卷积后大小不变
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg9_bn(**kwargs):
    '''
    my VGG 9-layer model with batch normalization
    '''
    model = VGG(make_layers(cfg, batch_norm=True), **kwargs)
    return model
