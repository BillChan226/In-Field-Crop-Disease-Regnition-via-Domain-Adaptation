"""
dataloader
"""
import h5py
from torch.utils.data import DataLoader, Dataset


class PlantDataSet(Dataset):
    '''
    训练集或者验证和测试数据
    '''
    def __init__(self, flag='train'):
        '''
        有些是通过图片的路径名称直接取得文件，这种通常针对与数据集太大读不进内存，
        这种 IO 会非常耗时。
        这里，数据集大概 12 GB， 内存完全够用，直接先读进来。
        '''
        train_path = '../dataset/plant_disease_a_0255/train_data.h5'
        val_path = '../dataset/plant_disease_a_0255/val_data.h5'
        test_path = '../dataset/plant_disease_a_0255/test_data.h5'
        self.file = None
        if flag == 'train':
            self.file = h5py.File(train_path, 'r')
        elif flag == 'val':
            self.file = h5py.File(val_path, 'r')
        else:
            self.file = h5py.File(test_path, 'r')
        self.x_data = self.file['X']
        self.y_data = self.file['Y']

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)

    def close_file(self):
        '''
        关闭文件
        '''
        self.file.close()


class PlantDataSetB(Dataset):
    '''
    训练集或者验证和测试数据
    '''
    def __init__(self, flag='train'):
        '''
        有些是通过图片的路径名称直接取得文件，这种通常针对与数据集太大读不进内存，
        这种 IO 会非常耗时。
        这里，数据集大概 12 GB， 内存完全够用，直接先读进来。
        目前定义，b 域为 5_20 噪声的数据集，19类
        '''
        train_path = '../dataset/plant_disease_b_0255/train_data_noise_5_20.h5'
        val_path = '../dataset/plant_disease_b_0255/val_data_noise_5_20.h5'
        test_path = '../dataset/plant_disease_b_0255/test_data_noise_5_20.h5'
        self.file = None
        if flag == 'train':
            self.file = h5py.File(train_path, 'r')
        elif flag == 'val':
            self.file = h5py.File(val_path, 'r')
        else:
            self.file = h5py.File(test_path, 'r')
        self.x_data = self.file['X']
        self.y_data = self.file['Y']

    def __getitem__(self, index):
        # b 域数据集直接储存的是 float 类型，无需归一化
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)

    def close_file(self):
        '''
        关闭文件
        '''
        self.file.close()


if __name__ == '__main__':
    TRAIN_DATA = PlantDataSet(flag='test')
    TRAIN_LOADER = DataLoader(TRAIN_DATA, batch_size=4, shuffle=True)
    print(type(TRAIN_LOADER))
    TRAIN_DATA_ITER = iter(TRAIN_LOADER)
    IMG, LABEL = TRAIN_DATA_ITER.next()
    print(type(IMG))
    print(IMG.shape)
    print(LABEL.shape)
    #print(LABEL)
    TRAIN_DATA.close_file()
