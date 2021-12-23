'''
模型在不同数据集下的预测结果
'''

from inspect import indentsize
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
from sklearn.metrics import accuracy_score  # 精度
from sklearn.metrics import confusion_matrix  # 混淆矩阵
from sklearn.metrics import classification_report
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, Subset

from model.inceptionv4 import inceptionv4
from model.mobilenetv2 import mobilenetv2
from model.resnet import resnet18
from model.shufflenetv2 import shufflenetv2
from model.vgg import vgg9_bn
from s3_dataset import PlantDataSet, PlantDataSetB


def get_pre(net, device, data_loader):
    '''
    得到整个测试集预测的结果，以及标签
    '''
    label_all = []
    pre_all = []
    with torch.no_grad():
        net.eval()
        for data in data_loader:
            images, labels = data
            images = images.float().to(device)
            labels = labels.long().to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            label_all.extend(labels.data.cpu().numpy())
            pre_all.extend(predicted.data.cpu().numpy())
    return pre_all, label_all


def save_file(save_path, y_pre, y_true):
    '''
    保存数据文件
    '''
    with h5py.File(save_path, 'w') as f_name:
        f_name['y_pre'] = y_pre
        f_name['y_true'] = y_true


# 5个原始模型在两个域上的测试集上的预测结果
Device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
Func = [vgg9_bn, resnet18, shufflenetv2, mobilenetv2, inceptionv4]

# 两个模型
Save_path_original = [
    '../model_save/plant_disease2/vgg.pth',
    '../model_save/plant_disease2/resnet18.pth',
    '../model_save/plant_disease2/shufflenetv2.pth',
    '../model_save/plant_disease2/mobilenetv2.pth',
    '../model_save/plant_disease2/inceptionv4.pth'
]
Save_path_domain = [
    '../model_save/plant_domain_temp/vgg.pth',
    '../model_save/plant_domain_temp/resnet18.pth',
    '../model_save/plant_domain_temp/shufflenetv2.pth',
    '../model_save/plant_domain_temp/mobilenetv2.pth',
    '../model_save/plant_domain_temp/inceptionv4.pth'
]

Save_path_old = [
    '../model_save/plant_disease/vgg.pth',
    '../model_save/plant_disease/resnet18.pth',
    '../model_save/plant_disease/shufflenetv2.pth',
    '../model_save/plant_disease/mobilenetv2.pth',
    '../model_save/plant_disease/inceptionv4.pth'
]


def test_model(model_paths, data_val, data_test, model_type, data_type):
    '''
    测试模型
    '''
    model_name = [
        'vgg', 'resnet', 'shufflenetv2', 'mobilenetv2', 'inceptionv4'
    ]
    for Index in range(5):
        net = Func[Index]()
        path_saved_model = model_paths[Index]
        net.load_state_dict(torch.load(path_saved_model))
        net.to(Device)
        # 验证集
        pre, label = get_pre(net, Device, data_val)
        pre, label = np.array(pre), np.array(label)
        print('{:s} 模型 {:s} 在 {:s} 域验证集上的预测精度为：{:.9f}'.format(
            model_type, model_name[Index], data_type,
            accuracy_score(label, pre)))
        # 保存预测结果
        path = '../result/s3/{:s}_{:s}_{:s}_val.h5'.format(
            model_type, model_name[Index], data_type)
        save_file(path, y_pre=pre, y_true=label)
        # 测试集
        pre, label = get_pre(net, Device, data_test)
        pre, label = np.array(pre), np.array(label)
        print('{:s} 模型 {:s} 在 {:s} 域测试集上的预测精度为：{:.9f}'.format(
            model_type, model_name[Index], data_type,
            accuracy_score(label, pre)))
        # 保存预测结果
        path = '../result/s3/{:s}_{:s}_{:s}_test.h5'.format(
            model_type, model_name[Index], data_type)
        save_file(path, y_pre=pre, y_true=label)


if __name__ == '__main__':
    # 两个数据集
    data_a_val = DataLoader(PlantDataSet(flag='val'),
                            batch_size=16,
                            shuffle=False)
    data_a_test = DataLoader(PlantDataSet(flag='test'),
                             batch_size=16,
                             shuffle=False)
    data_b_val = DataLoader(PlantDataSetB(flag='val'),
                            batch_size=16,
                            shuffle=False)
    data_b_test = DataLoader(PlantDataSetB(flag='test'),
                             batch_size=16,
                             shuffle=False)
    # test_model(Save_path_original, data_a_val, data_a_test, 'original', 'A')
    # test_model(Save_path_original, data_b_val, data_b_test, 'original', 'B')
    test_model(Save_path_domain, data_a_val, data_a_test, 'domain', 'A')
    test_model(Save_path_domain, data_b_val, data_b_test, 'domain', 'B')
    

# 模型的结果
# =====================================================================
# original 模型 vgg 在 A 域验证集上的预测精度为：0.965285682
# original 模型 vgg 在 A 域测试集上的预测精度为：0.964716595
# original 模型 resnet 在 A 域验证集上的预测精度为：0.986341908
# original 模型 resnet 在 A 域测试集上的预测精度为：0.988276804
# original 模型 shufflenetv2 在 A 域验证集上的预测精度为：0.988732074
# original 模型 shufflenetv2 在 A 域测试集上的预测精度为：0.988276804
# original 模型 mobilenetv2 在 A 域验证集上的预测精度为：0.984862281
# original 模型 mobilenetv2 在 A 域测试集上的预测精度为：0.985545186
# original 模型 inceptionv4 在 A 域验证集上的预测精度为：0.972114728
# original 模型 inceptionv4 在 A 域测试集上的预测精度为：0.972000911
# original 模型 vgg 在 B 域验证集上的预测精度为：0.803968254
# original 模型 vgg 在 B 域测试集上的预测精度为：0.815226011
# original 模型 resnet 在 B 域验证集上的预测精度为：0.848412698
# original 模型 resnet 在 B 域测试集上的预测精度为：0.849325932
# original 模型 shufflenetv2 在 B 域验证集上的预测精度为：0.369841270
# original 模型 shufflenetv2 在 B 域测试集上的预测精度为：0.386994449
# original 模型 mobilenetv2 在 B 域验证集上的预测精度为：0.189682540
# original 模型 mobilenetv2 在 B 域测试集上的预测精度为：0.196669310
# original 模型 inceptionv4 在 B 域验证集上的预测精度为：0.622222222
# original 模型 inceptionv4 在 B 域测试集上的预测精度为：0.621728787
# domain 模型 vgg 在 A 域验证集上的预测精度为：0.944684726
# domain 模型 vgg 在 A 域测试集上的预测精度为：0.944001821
# domain 模型 resnet 在 A 域验证集上的预测精度为：0.978260870
# domain 模型 resnet 在 A 域测试集上的预测精度为：0.979057592
# domain 模型 shufflenetv2 在 A 域验证集上的预测精度为：0.972228545
# domain 模型 shufflenetv2 在 A 域测试集上的预测精度为：0.974049624
# domain 模型 mobilenetv2 在 A 域验证集上的预测精度为：0.750398361
# domain 模型 mobilenetv2 在 A 域测试集上的预测精度为：0.755292511
# domain 模型 inceptionv4 在 A 域验证集上的预测精度为：0.944115639
# domain 模型 inceptionv4 在 A 域测试集上的预测精度为：0.948782153
# domain 模型 vgg 在 B 域验证集上的预测精度为：0.904761905
# domain 模型 vgg 在 B 域测试集上的预测精度为：0.921490880
# domain 模型 resnet 在 B 域验证集上的预测精度为：0.909523810
# domain 模型 resnet 在 B 域测试集上的预测精度为：0.925455987
# domain 模型 shufflenetv2 在 B 域验证集上的预测精度为：0.876984127
# domain 模型 shufflenetv2 在 B 域测试集上的预测精度为：0.870737510
# domain 模型 mobilenetv2 在 B 域验证集上的预测精度为：0.186507937
# domain 模型 mobilenetv2 在 B 域测试集上的预测精度为：0.193497224
# domain 模型 inceptionv4 在 B 域验证集上的预测精度为：0.843650794
# domain 模型 inceptionv4 在 B 域测试集上的预测精度为：0.876288660
# =====================================================================
