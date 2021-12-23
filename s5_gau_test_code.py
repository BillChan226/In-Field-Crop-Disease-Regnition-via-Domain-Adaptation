'''
测试模型运行速度和占用内存
'''
import os
import numpy as np
import skimage.io
import skimage.morphology
from skimage import io
from skimage.segmentation import mark_boundaries, slic
from skimage.util import img_as_float
from sklearn.mixture import GaussianMixture


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


def main():
    test_path = '../dataset/plant_seg/test'
    img = None
    img_path = os.path.join(test_path, 'img', str(4) + '.png')
    img = skimage.io.imread(img_path)  # 原始图片
    img = img.astype(np.float) / 255.0

    segments = slic(img, n_segments = 1000, sigma = 3)
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

    # 膨胀一下
    selem = skimage.morphology.disk(1)
    img_mask = skimage.morphology.dilation(img_mask, selem=selem)

if __name__ == '__main__':
    for i in range(10):
        main()
