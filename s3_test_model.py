# %%
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.inceptionv4 import inceptionv4
from model.mobilenetv2 import mobilenetv2
from model.resnet import resnet18
from model.shufflenetv2 import shufflenetv2
from model.vgg import vgg9_bn
from s3_dataset import PlantDataSet, PlantDataSetB

# %%
def get_acc(net, device, data_loader):
    '''
    get acc
    '''
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for data in data_loader:
            images, labels = data
            images = images.float().to(device)
            labels = labels.long().to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

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


# %%
Func = [vgg9_bn, resnet18, shufflenetv2, mobilenetv2, inceptionv4]
Save_path = [
    '../model_save/plant_disease2/vgg.pth',
    '../model_save/plant_disease2/resnet18.pth',
    '../model_save/plant_disease2/shufflenetv2.pth',
    '../model_save/plant_disease2/mobilenetv2.pth',
    '../model_save/plant_disease2/inceptionv4.pth'
]

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# data_loader_val = DataLoader(PlantDataSetB(flag='val'),
#                             batch_size=64,
#                             shuffle=False)
# data_loader_test = DataLoader(PlantDataSetB(flag='test'),
#                             batch_size=64,
#                             shuffle=False)

data_loader_val = DataLoader(PlantDataSet(flag='val'),
                             batch_size=64,
                             shuffle=False)
data_loader_test = DataLoader(PlantDataSet(flag='test'),
                              batch_size=64,
                              shuffle=False)

print('A 域数据集： 校核')
for Index in range(1):
    # 导入模型和权重
    net = Func[Index]()
    path_saved_model = Save_path[Index]
    net.load_state_dict(torch.load(path_saved_model))
    net.to(device)
    val_acc = get_acc(net, device, data_loader_val)
    test_acc = get_acc(net, device, data_loader_test)

    print('{:d}: val_acc:{:.5f}, test_acc:{:.5f}'.format(
        Index, val_acc, test_acc))

# %%
# 计算每个模型在两个测试集上的混淆矩阵
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
Func = [vgg9_bn, resnet18, shufflenetv2, mobilenetv2, inceptionv4]
Save_path = [
    '../model_save/plant_disease2/vgg.pth',
    '../model_save/plant_disease2/resnet18.pth',
    '../model_save/plant_disease2/shufflenetv2.pth',
    '../model_save/plant_disease2/mobilenetv2.pth',
    '../model_save/plant_disease2/inceptionv4.pth'
]
data_test_a = DataLoader(PlantDataSet(flag='test'),
                             batch_size=64,
                             shuffle=False)
data_test_b = DataLoader(PlantDataSetB(flag='test'),
                              batch_size=64,
                              shuffle=False)
Index = 1
# 导入模型和权重
net = Func[Index]()
path_saved_model = Save_path[Index]
net.load_state_dict(torch.load(path_saved_model))
net.to(device)
pre, label = get_pre(net, device, data_test_b)
pre, label = np.array(pre), np.array(label)

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score # 精度
from sklearn.metrics import confusion_matrix # 混淆矩阵

print('预测精度为：{:.9f}'.format(accuracy_score(label, pre)))
# 查看混淆矩阵
domain_A_class = {
    'Apple___Apple_scab': 0,
    'Apple___Black_rot': 1,
    'Apple___Cedar_apple_rust': 2,
    'Apple___healthy': 3,
    'Blueberry___healthy': 4,
    'Cherry_(including_sour)___Powdery_mildew': 5,
    'Cherry_(including_sour)___healthy': 6,
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7,
    'Corn_(maize)___Common_rust_': 8,
    'Corn_(maize)___Northern_Leaf_Blight': 9,
    'Corn_(maize)___healthy': 10,
    'Grape___Black_rot': 11,
    'Grape___Esca_(Black_Measles)': 12,
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)':13,
    'Grape___healthy':14,
    'Orange___Haunglongbing_(Citrus_greening)':15,
    'Peach___Bacterial_spot':16,
    'Peach___healthy':17,
    'Pepper,_bell___Bacterial_spot':18,
    'Pepper,_bell___healthy':19,
    'Potato___Early_blight':20,
    'Potato___Late_blight':21,
    'Potato___healthy':22,
    'Raspberry___healthy':23,
    'Soybean___healthy':24,
    'Squash___Powdery_mildew':25,
    'Strawberry___Leaf_scorch':26,
    'Strawberry___healthy':27,
    'Tomato___Bacterial_spot':28,
    'Tomato___Early_blight':29,
    'Tomato___Late_blight':30,
    'Tomato___Leaf_Mold':31,
    'Tomato___Septoria_leaf_spot':32,
    'Tomato___Spider_mites Two-spotted_spider_mite':33,
    'Tomato___Target_Spot':34,
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus':35,
    'Tomato___Tomato_mosaic_virus':36,
    'Tomato___healthy':37}

c_matrix = confusion_matrix(label, pre, labels=list(range(38)))


# %% 这个代码留着
def plot_Matrix(cm, classes, title=None,  cmap=plt.cm.Blues):
    plt.rc('font',family='Times New Roman',size='8')   # 设置字体样式、大小
    
    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    str_cm = cm.astype(np.str).tolist()
    for row in str_cm:
        print('\t'.join(row))
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j]*100 + 0.5) == 0:
                cm[i, j]=0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j]*100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j]*100 + 0.5) , fmt) + '%',
                        ha="center", va="center",
                        color="white"  if cm[i, j] > thresh else "black")
    
    
    plt.show()


# %%
domain_A_class.keys()

# %%
plt.matshow(cm, cmap=plt.cm.Blues)

# %%
