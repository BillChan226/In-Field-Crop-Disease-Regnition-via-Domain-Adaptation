'''
训练神经网络
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.vgg import vgg9_bn
from model.resnet import resnet18
from model.shufflenetv2 import shufflenetv2
from model.mobilenetv2 import mobilenetv2
from model.inceptionv4 import inceptionv4
from s3_dataset import PlantDataSet


BATCH_SIZE = 16


def get_val_acc(net, device):
    '''
    get val_acc
    '''
    # 导入验证集数据
    val_loader = DataLoader(PlantDataSet(flag='val'),
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
    val_loader = DataLoader(PlantDataSet(flag='test'),
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
    # 定义数据迭代器, 根据显存调整 batch_size 16
    train_loader = DataLoader(PlantDataSet(flag='train'),
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(10):
        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(train_loader, 0):
            imgs, labels = data
            imgs, labels = imgs.float().to(device), labels.long().to(device)
            optimizer.zero_grad()
            outputs = net(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # loss 大小
            running_loss += loss.item()
            # 得到训练集的预测精度
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # 打印信息 每 100 个 batch 打印一次
            if i % 100 == 99:
                fp.writelines('epoch:{:d}  num_batch:{:5d}  loss:{:.5f},  '
                      'train_acc:{:.5f}\n'.format(epoch + 1, i + 1,
                                                running_loss / 100,
                                                correct / total))
                fp.flush()
                running_loss = 0.0
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
    # 终端命令：nohup python -u s3_train.py > ../model_save/plant_disease/vgg_result.txt
    # 定义 GPU
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(device, ' is available!')
    # VGG, Resnet, Inception Net v , DenseNet, inceptionv4, ours
    Func = [vgg9_bn, resnet18,  shufflenetv2, mobilenetv2, inceptionv4]
    Save_path = ['../model_save/plant_disease_a/vgg.pth',
                 '../model_save/plant_disease_a/resnet18.pth',
                 '../model_save/plant_disease_a/shufflenetv2.pth',
                 '../model_save/plant_disease_a/mobilenetv2.pth',
                 '../model_save/plant_disease_a/inceptionv4.pth']
    Files_path = ['../model_save/plant_disease_a/vgg_result.txt',
                  '../model_save/plant_disease_a/resnet18_result.txt',
                  '../model_save/plant_disease_a/shufflenetv2_result.txt',
                  '../model_save/plant_disease_a/mobilenetv2_result.txt',
                  '../model_save/plant_disease_a/inceptionv4_result.txt']
    for Index in range(5):
        net = Func[Index]()
        net.to(device)
        train(net, device, Files_path[Index])
        # 保存模型
        torch.save(net.state_dict(), Save_path[Index])
