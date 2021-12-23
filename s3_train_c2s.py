'''
训练神经网络
域适应版本
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, BatchSampler

from model.vgg import vgg9_bn
from model.resnet import resnet18
from model.shufflenetv2 import shufflenetv2
from model.mobilenetv2 import mobilenetv2
from model.inceptionv4 import inceptionv4
from s3_dataset import PlantDataSet
from s3_dataset import PlantDataSetB
from mmd_loss import MMD_loss

BATCH_SIZE = 16


def get_val_acc(net, device):
    '''
    get val_acc
    '''
    # 导入验证集数据
    val_loader = DataLoader(PlantDataSetB(flag='val'),
                            batch_size=BATCH_SIZE,
                            shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for data in val_loader:
            images, labels = data
            images = images.float().to(device)
            labels = labels.long().to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    net.train()
    return correct / total


def get_test_acc(net, device):
    '''
    get test_acc
    '''
    # 导入验证集数据
    val_loader = DataLoader(PlantDataSetB(flag='test'),
                            batch_size=BATCH_SIZE,
                            shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for data in val_loader:
            images, labels = data
            images = images.float().to(device)
            labels = labels.long().to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    net.train()
    return correct / total


def train(net, device, file_path):
    fp = open(file_path, 'w')
    # source and target dataset
    source_set = PlantDataSet(flag='train')
    target_set = PlantDataSetB(flag='train')
    train_loader = DataLoader(source_set, batch_size=BATCH_SIZE, shuffle=True)
    random_sampler = RandomSampler(target_set,
                                   replacement=True,
                                   num_samples=len(source_set))
    batch_sampler = BatchSampler(random_sampler,
                                 batch_size=BATCH_SIZE,
                                 drop_last=False)
    train_target_loader = DataLoader(PlantDataSetB(flag='train'),
                                     batch_sampler=batch_sampler)
    # 优化器与交叉熵损失和 mmd 损失
    criterion = nn.CrossEntropyLoss()
    mmd = MMD_loss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # 训练循环
    for epoch in range(10):
        running_loss_cla = 0.0
        running_loss_mmd = 0.0
        total = 0
        correct = 0
        for i, data_combine in enumerate(
                zip(train_loader, train_target_loader), 0):
            # 从 dataloader 中获取数据
            data_source, data_target = data_combine
            imgs, labels = data_source
            imgs, labels = imgs.float().to(device), labels.long().to(device)

            imgs_target, _ = data_target
            imgs_target = imgs_target.float().to(device)

            # 梯度信息清空
            optimizer.zero_grad()
            # 前向传播，并得到 feature_map
            outputs = net(imgs)
            source_fmap = net.features_map
            # 目标域的特征层
            with torch.no_grad():
                # 根据 imgs_b 得到模型的中间层
                _ = net(imgs_target)
                targer_fmap = net.features_map

            # 根据 source_tensor 和 target_tensor 得到 MMD loss
            # 特征个数：vgg 16384, resnet 256, shufflenetv2 1024, mobilenet 512, inceptionv4 1536
            # vgg 效果肯定不好，最好的应该是resnet
            mmd_loss = mmd(source_fmap, targer_fmap)
            loss_cla = criterion(outputs, labels)
            loss = loss_cla + mmd_loss
            loss.backward()
            optimizer.step()
            # loss 大小
            running_loss_cla += loss_cla.item()
            running_loss_mmd += mmd_loss.item()
            # 得到训练集的预测精度
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # 打印信息 每 100 个 batch 打印一次
            if i % 100 == 99:
                fp.writelines(
                    'epoch:{:d}  num_batch:{:5d}  cla_loss:{:.5f},  mmd_loss:{:.5f}  train_acc:{:.5f}\n'
                    .format(epoch + 1, i + 1, running_loss_cla / 100,
                            running_loss_mmd / 100, correct / total))
                fp.flush()
                running_loss_cla = 0.0
                running_loss_mmd = 0.0
                correct = 0
                total = 0
        # 每一个 epoch 打印一次验证集预测精度
        val_acc = get_val_acc(net, device)
        fp.writelines('val_acc:{:.5f}\n'.format(val_acc))
        fp.flush()
    fp.writelines('Finished Training\n')
    fp.flush()
    fp.writelines('test_acc:{:.5f}\n'.format(get_test_acc(net, device)))
    fp.flush()
    fp.close()


if __name__ == '__main__':
    # 终端命令：nohup python -u s3_train_c2s.py > ../model_save/plant_disease/vgg_result.txt
    # 定义 GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device, ' is available!')
    # VGG, Resnet, Inception Net v , DenseNet, inceptionv4, ours
    Func = [vgg9_bn, resnet18, shufflenetv2, mobilenetv2, inceptionv4]
    Save_path = [
        '../model_save/plant_disease_domain2/vgg.pth',
        '../model_save/plant_disease_domain2/resnet18.pth',
        '../model_save/plant_disease_domain2/shufflenetv2.pth',
        '../model_save/plant_disease_domain2/mobilenetv2.pth',
        '../model_save/plant_disease_domain2/inceptionv4.pth'
    ]
    Files_path = [
        '../model_save/plant_disease_domain2/vgg_result.txt',
        '../model_save/plant_disease_domain2/resnet18_result.txt',
        '../model_save/plant_disease_domain2/shufflenetv2_result.txt',
        '../model_save/plant_disease_domain2/mobilenetv2_result.txt',
        '../model_save/plant_disease_domain2/inceptionv4_result.txt'
    ]
    for Index in [3]:
        net = Func[Index]()
        net.to(device)
        train(net, device, Files_path[Index])
        # 保存模型
        torch.save(net.state_dict(), Save_path[Index])
