'''
使用混合高斯聚类，对图片进行分割。
每张图片都是用一个混合高斯聚类模型。
超像素+混合高斯聚类？
'''
# %% [markdown]
# ## 读取数据集
#%% 先根据图片构建数据集
import numpy as np
from numpy.lib.function_base import append
import skimage.io
import skimage.morphology
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

#%%
def iu_acc(y_true, y_pred, th=0.5):
    smooth = 1e-12
    y_ = y_pred.copy()
    y_[y_ <= th] = 0
    y_[y_ > th] = 1
    inter = np.sum(y_pred * y_true)
    sum_ = np.sum(y_true + y_)
    return inter / (sum_ - inter + smooth)


def dice_acc(y_true, y_pred, th=0.5):
    smooth = 1e-12
    y_ = y_pred.copy()
    y_[y_ <= th] = 0
    y_[y_ > th] = 1
    inter = np.sum(y_ * y_true)
    sum_ = np.sum(y_true + y_)
    return 2 * inter / (sum_ + smooth)


def pre_recall(y_true, y_pred, th=0.5):
    y_ = y_pred.copy()
    y_[y_ <= th] = 0
    y_[y_ > th] = 1
    TP = np.sum(y_ * y_true)
    FP = np.sum(y_ * (1 - y_true))
    FN = np.sum(y_true * (1 - y_))
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    return P, R


def pixel_acc(y_true, y_pred):
    acc = np.sum(y_true == y_pred) / len(y_true)
    return acc
#%%
test_path = '../dataset/plant_seg/test'

test_data = []
test_label = []
for i in range(20):
    img_path = os.path.join(test_path, 'img', str(i) + '.png')
    ano_path = os.path.join(test_path, 'imgAno', str(i) + '.png')
    img = skimage.io.imread(img_path)  # 原始图片
    img_ano = skimage.io.imread(ano_path, 1)  # 灰度图
    # 归一化
    img = img.astype(np.float) / 255.0
    img_ano = img_ano.astype(np.float) / 255.0
    img_ano[img_ano > 0] = 1
    for i in range(0, 512):
        for j in range(0, 512):
            test_data.append(img[i, j, :])
            test_label.append(img_ano[i, j])

test_data = np.array(test_data)
test_label = np.array(test_label)

# %% [markdown]
# ## 构建混合高斯分布
fig, ax = plt.subplots(20, 2, figsize=(10, 5 * 20))
iu, dice,pixel,P,R,F2 = [],[],[],[],[],[]
for ii in range(20):
    model = GaussianMixture(n_components=3,
                            covariance_type='spherical',
                            max_iter=100,
                            random_state=0)

    model.fit(test_data[ii*512*512:(ii+1)*512*512])

    print('均值为：')
    print(model.means_)

    max_pos = np.argmax(np.linalg.norm(model.means_, axis=1))

    y_pre = model.predict(test_data)
    for j in range(3):
        if j != max_pos:
            y_pre[y_pre == j] = 5
        else:
            y_pre[y_pre == j] = 10
    y_pre[y_pre == 5] = 0
    y_pre[y_pre == 10] = 1
    
    img_pre = y_pre[ii * 512 * 512:(ii + 1) * 512 * 512]
    img_label = test_label[ii * 512 * 512:(ii + 1) * 512 * 512]

    img_or = test_data[ii * 512 * 512:(ii + 1) * 512 * 512, :]

    img_pre_dia = img_pre.reshape(512, 512)

    selem = skimage.morphology.disk(1)
    img_pre_dia = skimage.morphology.dilation(img_pre_dia, selem=selem)

    ax[ii][0].imshow(img_or.reshape(512, 512, 3))
    ax[ii][1].imshow(img_pre_dia, cmap='gray')

    img_pre_dia = img_pre_dia.reshape(-1)

    iu.append(iu_acc(img_label, img_pre_dia))
    dice.append(dice_acc(img_label, img_pre_dia))
    pixel.append(pixel_acc(img_label, img_pre_dia))
    p,r = pre_recall(img_label, img_pre_dia)
    f2 = 5*p*r/(4*p+r)
    P.append(p)
    R.append(r)
    F2.append(f2)

plt.show()
#%%
print('iu:', np.mean(iu))
print('dice:', np.mean(dice))
print('pixel:', np.mean(pixel))
print('P:',np.mean(P))
print('R:' ,np.mean(R))
print('F2:', np.mean(F2))

#%%


#####################################################
# 
# 
#                    超像素看结果
# 
# 
#####################################################

# %%
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
from skimage.segmentation import mark_boundaries
import numpy as np
# %%
def getPartsfeature(img, segments):
    # 统计每个区域的均值和方差
    out = []
    for i in range(np.max(segments)):
        part = img[np.where(segments==i)]
        mu = np.mean(part, axis=0)
        sigma = np.std(part, axis=0)
        out.append(mu)
    return np.array(out, dtype=np.float)

def fix_y_pre(y_pre, max_pos):
    for j in range(3):
        if j != max_pos:
            y_pre[y_pre == j] = 5
        else:
            y_pre[y_pre == j] = 10
    y_pre[y_pre == 5] = 0
    y_pre[y_pre == 10] = 1
# %%
iu, dice,pixel,P,R,F2 = [],[],[],[],[],[]
fig, ax = plt.subplots(20, 3, figsize=(15, 5 * 20))
for index in range(20):
    img = test_data[index*512*512:(index+1)*512*512].reshape(512,512,3)
    segments = slic(img, n_segments = 10000, sigma = 3)
    parts_features = getPartsfeature(img, segments)
    model = GaussianMixture(n_components=3,
                            covariance_type='spherical',
                            max_iter=20,
                            random_state=0)
    model.fit(parts_features)

    max_pos = np.argmax(np.linalg.norm(model.means_, axis=1))
    y_pre = model.predict(parts_features)
    fix_y_pre(y_pre, max_pos)

    img_mask = np.zeros([512,512])
    for i in range(len(y_pre)):
        if y_pre[i] == 1:
            img_mask[np.where(segments==i)] = 1
    
    img_label = test_label[index * 512 * 512:(index + 1) * 512 * 512]

    # 膨胀一下
    selem = skimage.morphology.disk(1)
    img_mask = skimage.morphology.dilation(img_mask, selem=selem)

    ax[index][0].imshow(mark_boundaries(img, segments))
    ax[index][1].imshow(img_label.reshape(512, 512), cmap='gray')
    ax[index][2].imshow(img_mask, cmap='gray')

    plt.imsave('../result/s5/model_pre/{:s}.png'.format(str(index+1)),img_mask, cmap='gray')

    

    iu.append(iu_acc(img_label, img_mask.reshape(-1)))
    dice.append(dice_acc(img_label, img_mask.reshape(-1)))
    pixel.append(pixel_acc(img_label, img_mask.reshape(-1)))
    p,r = pre_recall(img_label, img_mask.reshape(-1))
    P.append(p)
    R.append(r)
    F2.append(5*p*r/(4*p+r))
plt.show()
print('iu:', np.mean(iu))
print('dice:', np.mean(dice))
print('pixel:', np.mean(pixel))
print('P:',np.mean(P))
print('R:' ,np.mean(R))
print('F2:', np.mean(F2))


#%%




# %%
plt.figure()
fig,ax = plt.subplots(1,2, figsize=(10,5))
ax[0].imshow(img)
ax[1].imshow(img_mask, cmap='gray')
plt.show()
# %%

