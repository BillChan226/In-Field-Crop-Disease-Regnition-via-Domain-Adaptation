# ============================================================================
# 分析plant village 数据集，类别情况
# ============================================================================
# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import h5py
from matplotlib.font_manager import FontProperties  
font_song = FontProperties(fname='../util/simsun/simsun.ttc')

# %%
def get_X_Y(path):
    with h5py.File(path, 'r') as f_name:
        yy = f_name['Y'] 
        # 统计 
        dic = {}
        for c in yy:
            if c in dic.keys():
                dic[c] += 1
            else:
                dic[c] = 1
    X = sorted(dic.keys())
    Y = [dic[k] for k in X]
    return X, Y

#%%
path_train = '../dataset/plant_disease_a/train_data.h5'
X1,Y1 = get_X_Y(path_train)
path_val = '../dataset/plant_disease_a/val_data.h5'
X2,Y2 = get_X_Y(path_val)
path_test = '../dataset/plant_disease_a/test_data.h5'
X3,Y3 = get_X_Y(path_test)
Y_all = np.array(Y1) + np.array(Y2) + np.array(Y3)
#%%

#%%
plt.figure(figsize=(12, 5))

plt.bar(X1, Y1, color='deepskyblue', width=0.8, alpha=0.8, label='Train dataset')

plt.bar(X2, Y2,  color='orange',width=0.8, alpha=0.6, label='Val dataset', bottom=Y1)

plt.bar(X3, Y3,color='slategray' ,width=0.8, alpha=0.6, label='Test dataset', bottom=np.array(Y2)+np.array(Y1))

plt.xticks(list(range(38)))
plt.yticks(np.arange(0, 3800, 400))
plt.ylabel('数量',fontproperties=font_song, fontsize=18)
plt.xlabel('类别编号',fontproperties=font_song, fontsize=18)
plt.title('扩增版PlantVillages数据集类别分布',fontproperties=font_song, fontsize=18)
plt.legend()
for a, b in zip(X1, Y_all): 
    plt.text(a, b, '%d' % b, ha='center', va= 'bottom', rotation=45, fontsize=8) 
plt.savefig('../paper_result/plant_a_data_distribution.png', format='png', dpi=300, bbox_inches='tight')
plt.show()
#%%

# %%
# ============================================================================
# 绘制四个激活函数
# ============================================================================
import matplotlib.pyplot as plt
import numpy as np


# %%
# 激活函数
func_sigmoid = lambda x: 1/(1+np.exp(-x)) # sigmoid
func_tanh = lambda x: np.tanh(x) # tanh
func_relu = lambda x: np.maximum(0, x) # relu
# elu
def func_elu(x):
    out = []
    for i in x:
        c = i if i > 0 else np.exp(i) - 1
        out.append(c)
    return np.array(out)

def get_xy(func):
    x = np.arange(-10, 10, 0.1)
    y = func(x)
    return x,y

# %%
# sigmoid 激活函数
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 10))
fig1 = axes[0][0]
fig2 = axes[0][1]
fig3 = axes[1][0]
fig4 = axes[1][1]

x,y = get_xy(func_sigmoid)
fig1.plot(x, y, linewidth=3)
fig1.set_facecolor((0.95, 0.95, 0.95))
fig1.grid(color='white', linestyle='-', linewidth=2, alpha=1)
fig1.set_title('Sigmoid Activation Function', fontsize=16)
fig1.set_xlabel('x')
fig1.set_ylabel('y')

x,y = get_xy(func_tanh)
fig2.plot(x,y,linewidth=3)
fig2.set_facecolor((0.95, 0.95, 0.95))
fig2.grid(color='white', linestyle='-', linewidth=2, alpha=1)
fig2.set_title('Tanh Activation Function', fontsize=16)
fig2.set_xlabel('x')
fig2.set_ylabel('y')
x,y = get_xy(func_relu)
fig3.plot(x,y, linewidth=3)
fig3.set_facecolor((0.95, 0.95, 0.95))
fig3.grid(color='white', linestyle='-', linewidth=2, alpha=1)
fig3.set_title('Relu Activation Function', fontsize=16)
fig3.set_xlabel('x')
fig3.set_ylabel('y')
x,y = get_xy(func_elu)
fig4.plot(x,y, linewidth=3)
fig4.set_facecolor((0.95, 0.95, 0.95))
fig4.grid(color='white', linestyle='-', linewidth=2, alpha=1)
fig4.set_title('Elu Activation Function', fontsize=16)
fig4.set_xlabel('x')
fig4.set_ylabel('y')
plt.savefig('../paper_result/activation_func.png', format='png', dpi=300)
plt.show()


# %%
# ============================================================================
# 查看plant village 数据集中的图片
# ============================================================================

# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import h5py
import cv2 as cv
from matplotlib.font_manager import FontProperties  
font_song = FontProperties(fname='../util/simsun/simsun.ttc')

# %%
IMGS = np.zeros([25, 256, 256, 3], dtype=np.uint8)
path = '../for_plot/plant_a_sample/'
for i, pth in enumerate(sorted(os.listdir(path))):
    pt = os.path.join(path, pth)
    img = plt.imread(pt)
    IMGS[i] = img

# %%
fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
for i in range(5):
    for j in range(5):
        fig_ = ax[i][j]
        fig_.imshow(IMGS[i*5+j])
        fig_.set_xticks([])
        fig_.set_yticks([])
fig.subplots_adjust(wspace=0.005, hspace=0.05)
plt.savefig('../paper_result/show_imgs.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# ============================================================================
# 查看 复杂域数据集中的图片
# ============================================================================

# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import h5py
import cv2 as cv

#%%
def img_gaussian(img, mu = 5, sigma=20):
    '''
    给图片加上高斯噪声
    '''
    img_int16 = img.astype(np.int16)
    img_noise = img_int16 + (mu + np.random.randn(*img.shape) * sigma).astype(np.int16)
    img_noise = np.clip(img_noise, 0, 255)
    out = img_noise.astype(np.uint8)
    return out
#%%
# 得到10张复杂背景的图片 
IMGS = np.zeros([12, 256, 256, 3], dtype=np.uint8)
path = '../for_plot/plant_b/'
for i, pth in enumerate(sorted(os.listdir(path)), 0):
    pt = os.path.join(path, pth)
    img = cv.imread(pt)
    img = cv.resize(img, (256, 256))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 转化为 RGB 格式
    IMGS[i] = img_gaussian(img)

# %%
# 得到10张带有噪声的图片
IMGS2 = np.zeros([10, 256, 256, 3], dtype=np.uint8)
path = '../dataset/plant_disease_b_0255/val_data_noise_5_20.h5'
with h5py.File(path, 'r') as f_name:
    xx = f_name['X'] 
    for i in range(10):
        IMGS2[i, :,:,0] = xx[i, 0,:,:]
        IMGS2[i, :,:,1] = xx[i, 1,:,:]
        IMGS2[i, :,:,2] = xx[i, 2,:,:]
# %%
fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(10, 8))
for i in range(4):
    for j in range(5):
        fig_ = ax[i][j]
        fig_.set_xticks([])
        fig_.set_yticks([])
        if i*5+j < 10:
            fig_.imshow(IMGS[i*5+j+2])
        else:
            fig_.imshow(IMGS2[i*5+j-10])
        
fig.subplots_adjust(wspace=0.005, hspace=0.05)
plt.savefig('../paper_result/show_imgs2.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# ============================================================================
# 分析 复杂域 数据集，类别情况
# ============================================================================
# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import h5py
from matplotlib.font_manager import FontProperties
font_song = FontProperties(fname='../util/simsun/simsun.ttc')

# %%
def get_X_Y(path):
    with h5py.File(path, 'r') as f_name:
        yy = f_name['Y'] 
        # 统计 
        dic = {}
        for c in yy:
            if c in dic.keys():
                dic[c] += 1
            else:
                dic[c] = 1
    X = sorted(dic.keys())
    Y = [dic[k] for k in X]
    return X, Y

#%%
path_val = '../dataset/plant_disease_b_0255/val_data.h5'
X2,Y2 = get_X_Y(path_val)
path_test = '../dataset/plant_disease_b_0255/test_data.h5'
X3,Y3 = get_X_Y(path_test)
Y_all = np.array(Y2) + np.array(Y3)
#%%

#%%
plt.figure(figsize=(10, 5))
X = [str(i) for i in X2]
plt.bar(X, Y2,  color='deepskyblue',width=0.8, alpha=0.6, label='Val dataset')

plt.bar(X, Y3,color='orange' ,width=0.8, alpha=0.6, label='Test dataset', bottom=np.array(Y2))

plt.xticks(X)
plt.yticks(np.arange(0, 800, 100))
plt.ylabel('数量',fontproperties=font_song, fontsize=18)
plt.xlabel('类别编号',fontproperties=font_song, fontsize=18)
plt.title('复杂域数据集类别分布',fontproperties=font_song, fontsize=18)
plt.legend()
for a, b in zip(list(range(19)), Y_all): 
    plt.text(a, b, '%d' % b, ha='center', va= 'bottom', rotation=45, fontsize=8) 
plt.savefig('../paper_result/plant_b_data_distribution.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# ============================================================================
# 绘制 白粉病数据集 拍摄图片
# ============================================================================

# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import h5py
import cv2 as cv
from matplotlib.font_manager import FontProperties
font_song = FontProperties(fname='../util/simsun/simsun.ttc')

# %%
path = '../dataset/plant_seg/train/img'
imgs = np.zeros([25, 512, 512, 3], dtype=np.uint8)
for i in range(25):
    img = cv.imread(os.path.join(path, str(i)+'.png'))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 转化为 RGB 格式
    img[img==0] = 255
    imgs[i] = img
    

# %%
M = 4
N = 5
fig, ax = plt.subplots(nrows=M, ncols=N, figsize=(15,12))
for i in range(M):
    for j in range(N):
        fig_ = ax[i][j]
        fig_.set_xticks([])
        fig_.set_yticks([])
        fig_.imshow(imgs[i*N+j])
        
fig.subplots_adjust(wspace=0.003, hspace=0.03)
plt.savefig('../paper_result/img_seg_data_raw.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# ============================================================================
# 绘制 白粉病数据集 标注图片
# ============================================================================
#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import h5py
import cv2 as cv
from matplotlib.font_manager import FontProperties
font_song = FontProperties(fname='../util/simsun/simsun.ttc')


#%%
path_img = '../dataset/plant_seg/train/img'
path_ano = '../dataset/plant_seg/train/imgAno'
imgs = np.zeros([5, 512, 512, 3], dtype=np.uint8)
imgs_ano = np.zeros([5, 512, 512, 3], dtype=np.uint8)
imgs_d = np.zeros([5, 512, 512, 3], dtype=np.uint8)

for j, i in enumerate([7, 19, 21, 24, 25]):
    img = cv.imread(os.path.join(path_img, str(i)+'.png'))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 转化为 RGB 格式
    imgs[j] = img

    img = cv.imread(os.path.join(path_ano, str(i)+'.png'))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 转化为 RGB 格式
    imgs_ano[j] = img
    img[img!=0] = 255
    imgs_d[j] = img

# %%
M = 3
N = 5
fig, ax = plt.subplots(nrows=M, ncols=N, figsize=(20,12))
for j in range(N):
    fig_ = ax[0][j]
    fig_.set_xticks([])
    fig_.set_yticks([])
    fig_.imshow(imgs[j])

    fig_ = ax[1][j]
    fig_.set_xticks([])
    fig_.set_yticks([])
    fig_.imshow(imgs_ano[j])

    fig_ = ax[2][j]
    fig_.set_xticks([])
    fig_.set_yticks([])
    fig_.imshow(imgs_d[j])


        
fig.subplots_adjust(wspace=0.003, hspace=0.03)
plt.savefig('../paper_result/img_seg_dataset.png', format='png', dpi=400, bbox_inches='tight')
plt.show()

# %%
# ============================================================================
# 绘制6个表格的图片
# ============================================================================
# %%
