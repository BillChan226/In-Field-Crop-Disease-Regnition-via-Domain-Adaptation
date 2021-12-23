'''
用 GBDT 分割图片
'''
#%%
import lightgbm as lgb
import pandas as pd
import numpy as np
import skimage.io
import os
from sklearn.metrics import mean_squared_error

#%%
# load or create your dataset
print('Load data...')
# read data
train_path = '../dataset/plant_seg/train'
test_path = '../dataset/plant_seg/test'

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

# test data
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
#%%
# create dataset for lightgbm
lgb_train = lgb.Dataset(train_data, train_label)
lgb_test = lgb.Dataset(test_data, test_label, reference=lgb_train)

#%%
# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 2,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=50,
                valid_sets=lgb_test,
                early_stopping_rounds=5)


print('Start predicting...')
#%%
# predict
y_pred = gbm.predict(test_data, num_iteration=gbm.best_iteration)
# eval
#%%
y_pred_c = np.argmax(y_pred, axis = 1)
acc = np.sum((y_pred_c == test_label))/test_label.size
print(acc)

#%% 3 acc and pre and recall
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
    yy_pred = y_pred_c[i:i+262144]
    yy_label = test_label[i:i+262144]
    print(iu_acc(yy_label, yy_pred))
# dice
print('dice:')
for i in range(0, 5242880, 262144):
    yy_pred = y_pred_c[i:i+262144]
    yy_label = test_label[i:i+262144]
    print(dice_acc(yy_label, yy_pred))
# pixel
print('pixel:')
for i in range(0, 5242880, 262144):
    print(np.sum((y_pred_c[i:i+262144] == test_label[i:i+262144]))/262144)
    
#%%
print('pre')
# pre recall
for i in range(0, 5242880, 262144):
    yy_pred = y_pred_c[i:i+262144]
    yy_label = test_label[i:i+262144]
    ans = pre_recall(yy_label, yy_pred)
    print(ans[0])
#%%
print('recall')
# pre recall
for i in range(0, 5242880, 262144):
    yy_pred = y_pred_c[i:i+262144]
    yy_label = test_label[i:i+262144]
    ans = pre_recall(yy_label, yy_pred)
    print(ans[1])



#%% 保存预测图片
import matplotlib.pyplot as plt
test_result = y_pred_c.reshape(20, 512, 512)
for i in range(20):
    plt.figure()
    plt.imshow(test_result[i, :, :], cmap='gray')
    plt.imsave('.\\GBDT_pre_image\\' + str(i) + '.png', test_result[i, :, :], cmap='gray')

