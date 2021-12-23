# %% 
import numpy as np
import skimage.io
from matplotlib import pyplot as plt
import os

# %%
def rgb2gray(img):
    '''
    灰度化
    '''
    r = img[:, :, 0].copy().astype(np.float32)
    g = img[:, :, 1].copy().astype(np.float32)
    b = img[:, :, 2].copy().astype(np.float32)

    gray_img = 0.2126 * r + 0.7152 * g + 0.0722 * b
    gray_img = gray_img.astype(np.uint8)

    return gray_img
#
# %%
def otsu(img):
    '''
    otsu 算法
    '''
    gray_img = img.copy()
    h = gray_img.shape[0]
    w = gray_img.shape[1]
    threshold_t = 0
    max_g = 0
    
    # 遍历每一个灰度层
    for t in range(255):
    	# 使用numpy直接对数组进行运算
        n0 = gray_img[np.where(gray_img < t)]
        n1 = gray_img[np.where(gray_img >= t)]
        w0 = len(n0) / (h * w)
        w1 = len(n1) / (h * w)
        u0 = np.mean(n0) if len(n0) > 0 else 0.
        u1 = np.mean(n1) if len(n0) > 0 else 0.
        
        g = w0 * w1 * (u0 - u1) ** 2
        if g > max_g:
            max_g = g
            threshold_t = t
    print('类间方差最大阈值：', threshold_t)
    gray_img[gray_img < threshold_t] = 0
    gray_img[gray_img >= threshold_t] = 255
    return gray_img

def otsu_back(img):
    '''
    不考虑背景的 otsu 算法
    '''
    gray_img = img.copy()
    h = gray_img.shape[0]
    w = gray_img.shape[1]
    threshold_t = 0
    max_g = 0
    
    # 遍历每一个灰度层
    for t in range(1, 255):
    	# 使用numpy直接对数组进行运算
        n0 = gray_img[np.where((0 < gray_img) * (gray_img < t))]
        n1 = gray_img[np.where(gray_img >= t)]
        w0 = len(n0) / (h * w)
        w1 = len(n1) / (h * w)
        u0 = np.mean(n0) if len(n0) > 0 else 0.
        u1 = np.mean(n1) if len(n0) > 0 else 0.
        
        g = w0 * w1 * (u0 - u1) ** 2
        if g > max_g:
            max_g = g
            threshold_t = t
    # print('类间方差最大阈值：', threshold_t)
    gray_img[gray_img < threshold_t] = 0
    gray_img[gray_img >= threshold_t] = 255
    return gray_img

# %%
def iu_acc(y_true, y_pred, th = 0.5):
    smooth = 1e-12
    y_ = y_pred.copy()
    y_[y_ <= th] = 0
    y_[y_ > th] = 1
    inter = np.sum(y_pred * y_true)
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

def pixel_acc(y_true, y_pred):
    acc = np.sum(y_true == y_pred) / (y_true.shape[0]*y_true.shape[1])
    return acc
# %%
# 这里直接将数据转换成float32了，方便后续计算
path = '../dataset/plant_seg/test/img/4.png'
img = skimage.io.imread(path)
gray_img = rgb2gray(img)
otsu_img = otsu_back(gray_img)
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10,30))
ax[0].imshow(img)
ax[1].imshow(gray_img, cmap='gray')
ax[2].imshow(otsu_img, cmap='gray')
plt.show()
# %%
path = '../dataset/plant_seg/test/imgAno/4.png'
img_ano = skimage.io.imread(path, as_gray=True) 
# %%
path_img = '../dataset/plant_seg/test/img'
path_img_ano = '../dataset/plant_seg/test/imgAno'
path_save = '../result/s4/otsu/'
for i in range(20):
    pth_img = os.path.join(path_img, str(i)+'.png')
    pth_img_ano = os.path.join(path_img_ano, str(i)+'.png')
    img = (skimage.io.imread(pth_img, as_gray=True) * 255).astype(np.uint8)
    img_otsu = otsu_back(img)
    img_otsu = img_otsu.astype(np.float64) / 255.0
    img_ano = skimage.io.imread(pth_img_ano, as_gray=True)
    img_ano[img_ano > 0] = 1

    iou = iu_acc(img_ano, img_otsu)
    dice = dice_acc(img_ano, img_otsu)
    pre, recall = pre_recall(img_ano, img_otsu)
    pix_acc = pixel_acc(img_ano, img_otsu)

    print('{:.5f} {:.5f} {:.5f} {:.5f} {:.5f}&'.format(iou, dice, pix_acc, pre, recall))
    # 保存被分割的图片
    # plt.imsave(os.path.join(path_save, str(i)+'.png') , img_otsu, cmap='gray')




# %%
