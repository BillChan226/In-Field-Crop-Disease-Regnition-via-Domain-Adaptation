"""
构建 A 域数据集
读取图片，保存到 h5 文件中
保存的文件形状为:
    img: (num_img, 3, 256, 256)
    label: (num_img) // 不是onehot编码
"""
import os
import cv2 as cv
import h5py
import numpy as np


def read_imgs(x_data, y_data, file_dir, total_num):
    '''
    从文件夹中读取文件到 x_data, y_data。
    '''
    img_count = 0
    cla_count = 0
    for cla in sorted(os.listdir(file_dir)):  # 这里一定要排序
        for file_name in os.listdir(os.path.join(file_dir, cla)):
            print("\rreading...:{:d}/{:d}".format(img_count + 1, total_num),
                  end='')
            file_path = os.path.join(file_dir, cla, file_name)
            img = cv.imread(file_path)  # 默认是 BGR 格式
            # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 转化为 RGB 格式
            # 变为 (3, 256, 256) 的 RGB 图像
            img = np.array([img[:, :, 2], img[:, :, 1], img[:, :, 0]])
            x_data[img_count, :, :, :] = img
            y_data[img_count] = cla_count
            img_count += 1
        cla_count += 1


def save_file(save_path, x_data, y_data):
    '''
    保存数据文件
    '''
    with h5py.File(save_path, 'w') as f_name:
        f_name['X'] = x_data
        f_name['Y'] = y_data


def get_train_data_h5(file_dir, save_path, num=70295):
    '''
    得到训练集数据 h5 文件
    '''
    print("读取文件的路径为:{:s}".format(file_dir))
    # 读取文件
    x_data = np.zeros([num, 3, 256, 256], dtype=np.uint8)  # 有 num 张图片
    y_data = np.zeros([num], dtype=np.uint8)  # 有 38 个类别
    read_imgs(x_data, y_data, file_dir, num)  # 不用函数返回的原因是: 少一次拷贝
    # 保存为 h5 文件
    print("\nsaving...")
    save_file(save_path, x_data, y_data)
    print("finished!")


def get_val_and_test_data_h5(file_dir,
                             save_path_val,
                             save_path_test,
                             num=17572):
    '''
    得到集数据 h5 文件
    '''
    print("读取文件的路径为:{:s}".format(file_dir))
    # 读取文件
    x_data = np.zeros([num, 3, 256, 256], dtype=np.uint8)  # 有 num 张图片
    y_data = np.zeros([num], dtype=np.uint8)  # 有 38 个类别
    read_imgs(x_data, y_data, file_dir, num)  # 不用函数返回的原因是: 少一次拷贝
    # 随机打乱 x_data y_data, 固定随机数种子，保证每次都一样
    seq = list(range(num))
    np.random.seed(1)
    np.random.shuffle(seq)
    num_val = int(num / 2)
    num_test = int(num - num_val)
    # 构建验证数据
    x_val_data = np.zeros([num_val, 3, 256, 256], dtype=np.uint8)
    y_val_data = np.zeros([num_val], dtype=np.uint8)
    for i in range(num_val):
        x_val_data[i] = x_data[seq[i]]
        y_val_data[i] = y_data[seq[i]]
    # 构建测试数据
    x_test_data = np.zeros([num_test, 3, 256, 256], dtype=np.uint8)
    y_test_data = np.zeros([num_test], dtype=np.uint8)
    for i in range(num_test):
        x_test_data[i] = x_data[seq[i + num_val]]
        y_test_data[i] = y_data[seq[i + num_val]]
    # 保存为 h5 文件
    print("\nsaving...")
    save_file(save_path_val, x_val_data, y_val_data)
    save_file(save_path_test, x_test_data, y_test_data)
    print("finished!")


if __name__ == '__main__':
    # 一些路径
    READ = [
        '../raw_dataset/plant_data/train', '../raw_dataset/plant_data/valid'
    ]
    SAVE = [
        '../dataset/plant_disease_a_0255/train_data.h5',
        '../dataset/plant_disease_a_0255/val_data.h5',
        '../dataset/plant_disease_a_0255/test_data.h5'
    ]
    get_train_data_h5(READ[0], SAVE[0], 70295)
    get_val_and_test_data_h5(READ[1], SAVE[1], SAVE[2], 17572)
