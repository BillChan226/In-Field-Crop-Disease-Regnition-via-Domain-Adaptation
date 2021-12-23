
import numpy as np
import skimage.io
from sklearn import svm
import os

# read data
train_path = '.\\data\\train'
test_path = '.\\data\\test'
#%% train data 
train_data = []
train_label = []
for x in os.listdir(os.path.join(train_path, 'img')):
    img_path = os.path.join(train_path, 'img', x)
    ano_path = os.path.join(train_path, 'imgAno', x)
    img = skimage.io.imread(img_path)
    img_ano = skimage.io.imread(ano_path, 1)
    # 归一化
    img = img.astype(np.float)/255
    img_ano = img_ano.astype(np.float)/255
    img_ano[img_ano > 0] = 1
    for i in range(0, 512):
        for j in range(0, 512):
            train_data.append(img[i, j, :])
            train_label.append(img_ano[i, j])

train_data = np.array(train_data)
train_label = np.array(train_label)



# shuffle data
train = np.hstack((train_data, train_label.reshape(-1, 1)))
np.random.shuffle(train)
train_data = train[:, 0:3]
train_label = train[:, 3]


#%% test data
test_data = []
test_label = []
for i in range(20):
    img_path = os.path.join(test_path, 'img', str(i) + '.png')
    ano_path = os.path.join(test_path, 'imgAno', str(i) + '.png')
    img = skimage.io.imread(img_path)
    img_ano = skimage.io.imread(ano_path, 1)
    # 归一化
    img = img.astype(np.float)/255
    img_ano = img_ano.astype(np.float)/255
    img_ano[img_ano > 0] = 1
    for i in range(0, 512):
        for j in range(0, 512):
            test_data.append(img[i, j, :])
            test_label.append(img_ano[i, j])

test_data = np.array(test_data)
test_label = np.array(test_label)


#%% segmentation using random forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10, verbose = True)
clf.fit(train_data, train_label)
#%%
y_pred = clf.predict(test_data)
#%% 
acc = np.sum((y_pred == test_label))/test_label.size
print(acc)
#%%
for i in range(0, 5242880, 262144):
    print(np.sum((y_pred[i:i+262144] == test_label[i:i+262144]))/262144)    
#%%
# 3个acc
def iu_acc(y_true, y_pred, th = 0.5):
    smooth = 1e-12
    y_ = y_pred.copy()
    y_[y_ <= th] = 0
    y_[y_ > th] = 1
    inter = np.sum(y_ * y_true)
    sum_ = np.sum(y_true + y_)
    return inter / (sum_ - inter + smooth) 

def dice_acc(y_true, y_pred, th = 0.5):
    smooth = 1e-12
    y_ = y_pred.copy()
    y_[y_ <= th] = 0
    y_[y_ > th] = 1
    inter = np.sum(y_ * y_true)
    sum_ = np.sum(y_true + y_)
    return 2 * inter / (sum_ + smooth) 

def pixel_acc(y_true, y_pred, th = 0.5):
    y_ = y_pred.copy()
    y_[y_ <= th] = 0
    y_[y_ > th] = 1
    inter = np.sum(y_ * y_true)
    inter2 = np.sum((1 - y_) * (1 - y_true))
    return (inter + inter2) / (np.size(y_, 0) * np.size(y_, 1))

def pre_recall(y_true, y_pred, th = 0.5):
    y_ = y_pred.copy()
    y_[y_ <= th] = 0
    y_[y_ > th] = 1
    TP = np.sum(y_ * y_true)
    FP = np.sum(y_ * (1 - y_true))
    FN = np.sum(y_true * (1 - y_))
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    return P, R
#%%
# iu
print('iu:')
for i in range(0, 5242880, 262144):
    yy_pred = y_pred[i:i+262144]
    yy_label = test_label[i:i+262144]
    print(iu_acc(yy_label, yy_pred))
# dice
print('dice:')
for i in range(0, 5242880, 262144):
    yy_pred = y_pred[i:i+262144]
    yy_label = test_label[i:i+262144]
    
    print(dice_acc(yy_label, yy_pred))
# pixel
print('pixel:')
for i in range(0, 5242880, 262144):
    print(np.sum((y_pred[i:i+262144] == test_label[i:i+262144]))/262144)
    
#%%
print('pre')
# pre recall
for i in range(0, 5242880, 262144):
    yy_pred = y_pred[i:i+262144]
    yy_label = test_label[i:i+262144]
    ans = pre_recall(yy_label, yy_pred)
    print(ans[0])
#%%
print('recall')
# pre recall
for i in range(0, 5242880, 262144):
    yy_pred = y_pred[i:i+262144]
    yy_label = test_label[i:i+262144]
    ans = pre_recall(yy_label, yy_pred)
    print(ans[1])


#%% 保存预测图片
import matplotlib.pyplot as plt
test_result = y_pred.reshape(20, 512, 512)
for i in range(20):
    plt.figure()
    plt.imshow(test_result[i, :, :], cmap='gray')
    plt.imsave('.\\RF_image\\' + str(i) + '.png', test_result[i, :, :], cmap='gray')



#%%
for i in range(20):
    plt.figure()
    plt.imshow(test_data.reshape(20, 512,512,3)[19 - i,:,:,:])
