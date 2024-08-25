import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as io
from sklearn.model_selection import StratifiedKFold

class GDataset(Dataset):
    def __init__(self, idxs, csv,  node_num):
        super(GDataset, self).__init__()
        self.fc_matrix_dot = '/mnt/data/zhangxin/Journal/data_for_local/npydata'
        self.csv = csv
        self.idxs = idxs
        # self.class_num = class_num
        # self.feature_mean = feature_mean
        # self.feature_std = feature_std
        self.node_num = node_num

    def __len__(self):
        return self.idxs.shape[0]

    def __getitem__(self, item):
        index = self.idxs[item]
        dir_name = self.csv['name'].iloc[index]
        label = self.csv['label'].iloc[index].astype(np.int)
        label_1 = self.csv['patch_1'].iloc[index].astype(np.int)
        label_2 = self.csv['patch_2'].iloc[index].astype(np.int)
        label_3 = self.csv['patch_3'].iloc[index].astype(np.int)
        label_4 = self.csv['patch_4'].iloc[index].astype(np.int)
        label_5 = self.csv['patch_5'].iloc[index].astype(np.int)
        fc_matrix_path = os.path.join(self.fc_matrix_dot, dir_name, f'fcmatrix_{self.node_num}.npy')
        fc_matrix = np.load(fc_matrix_path).astype(np.float32)
        #fc_matrix = 1 - np.sqrt((1 - fc_matrix) / 2)

        feature_path = os.path.join(self.fc_matrix_dot, dir_name, f'feature_{self.node_num}.npy')
        feature = np.load(feature_path).astype(np.float32)
        #feature = ((feature - self.feature_mean) / self.feature_std).astype(np.float32)[:, :3]
        #feature = ((feature - self.feature_mean) / self.feature_std).astype(np.float32)
        # feature_path = os.path.join(self.fc_matrix_dot, dir_name, 'ROISignals_feature.npy')
        # feature = np.load(feature_path).astype(np.float32)
        # feature_mean = np.array([[0.00304313, 2.71610968, 0.01176067]])
        # feature_std = np.array([[0.02716032, 3.4016537, 0.00158216405]])
        # feature = ((feature - feature_mean) / feature_std).astype(np.float32)

        # features = np.ones([116, 1], dtype=np.float32)
        return torch.tensor(fc_matrix), torch.tensor(feature), torch.tensor(label), torch.tensor(label_1),torch.tensor(label_2),torch.tensor(label_3),torch.tensor(label_4),torch.tensor(label_5)

class GTestDataset(Dataset):
    def __init__(self, idxs, csv, node_num):
        super(GTestDataset, self).__init__()
        self.fc_matrix_dot = os.path.join('test_0120_0950')
        # self.fc_matrix_dot = os.path.join('test_input_200_20_095')
        self.csv = csv
        self.idxs = idxs
        self.node_num = node_num

    def __len__(self):
        return self.idxs.shape[0]

    def __getitem__(self, item):
        index = self.idxs[item]
        dir_name = self.csv['name'].iloc[index]
        label = self.csv['label'].iloc[index].astype(np.int)
        risk = self.csv['risk'].iloc[index].astype(np.int)
        fc_matrix_path = os.path.join(self.fc_matrix_dot, dir_name, f'fcmatrix_{self.node_num}.npy')
        fc_matrix = np.load(fc_matrix_path).astype(np.float32)
        fc_matrix = 1 - np.sqrt((1 - fc_matrix) / 2)

        feature_path = os.path.join(self.fc_matrix_dot, dir_name, f'feature_{self.node_num}.npy')
        feature = np.load(feature_path).astype(np.float32)
        #feature = ((feature - self.feature_mean) / self.feature_std).astype(np.float32)

        # feature_path = os.path.join(self.fc_matrix_dot, dir_name, 'ROISignals_feature.npy')
        # feature = np.load(feature_path).astype(np.float32)
        # feature_mean = np.array([[0.00304313, 2.71610968, 0.01176067]])
        # feature_std = np.array([[0.02716032, 3.4016537, 0.00158216405]])
        # feature = ((feature - feature_mean) / feature_std).astype(np.float32)

        # features = np.ones([116, 1], dtype=np.float32)
        return torch.tensor(fc_matrix), torch.tensor(feature), torch.tensor(label), torch.tensor(risk),dir_name
def get_data_loader(node_num,bs):
    df = pd.read_csv('/mnt/data/zhangxin/Journal/data_for_local/data_csv/train_patch_0715.csv',encoding='utf-8')
    '''
    if class_name in ['one_vs_two']:
        extrelabel = 0
        df = df[df['label'] != extrelabel]
        df['label'] = df['label'] - 1
    elif class_name in ['zero_vs_two']:
        extrelabel = 1
        df = df[df['label'] != extrelabel]
        df.loc[df['label'] == 2, 'label'] = df.loc[df['label'] == 2, 'label'] - 1
    elif class_name in ['zero_vs_one']:
        extrelabel = 2
        df = df[df['label'] != extrelabel]
    '''
    df = df.reset_index()
    #print(df.shape)
    train_idxs = np.where((df['fold'] != 0))[0]
    val_idxs = np.where(df['fold']  == 0)[0]
    # train_idxs = np.where((df['fold'] != 3)&(df['fold']!=4)&(df['fold']!=5)&(df['fold']!=6))[0]
    # val_idxs = np.where((df['fold']  == 3)|(df['fold']==4)|(df['fold']==5))[0]
    '''
    train_idxs0 = np.where(df['fold'] != i_fold)[0]
    train_idxs1 = train_idxs0
    print(train_idxs0)
    y = df['label'][np.where(df['fold'] != i_fold)[0]]
    print(y)
    folder = StratifiedKFold(n_splits=5, random_state=12, shuffle=True)
    for train_index, val_index in folder.split(train_idxs1, y):
        print(train_index)
        print(val_index)
        train_idxs = train_idxs0[train_index]
        val_idxs = train_idxs0[val_index]
        break
    test_idxs = np.where(df['fold'] == i_fold)[0]
    '''
    ## featuremean featurestd to normalization
    # feature_list = []
    # for i, row in df.iterrows():
    #     name = row['name']
    #     feature_path = os.path.join('adni', 'adni_input_weiquchu', name, f'feature_{node_num}.npy')
    #     feature = np.load(feature_path).astype(np.float32)
    #     feature_list.append(feature)
    # print(np.array(feature_list).shape)
    # feature_mean = np.array(np.concatenate(feature_list).mean(axis=0))
    # feature_std = np.array(np.concatenate(feature_list).std(axis=0))
    # print(feature_mean)
    # print(feature_std)
    TrainDataset = GDataset(train_idxs, df, node_num)
    ValDataset = GDataset(val_idxs, df, node_num)
    TrainLoader = DataLoader(TrainDataset, batch_size=bs, shuffle=True)
    ValLoader = DataLoader(ValDataset, batch_size=1)
    return TrainLoader, ValLoader

def get_test_dataloader(node_num,batch_size,data_path):
    df = pd.read_csv(data_path)
    df = df.reset_index()
    #print(df.shape)

    test_idxs=np.where(df['name'] != None)[0]
    TestDataset = GTestDataset(test_idxs,df,node_num)
    TestLoader = DataLoader(TestDataset,batch_size=1)
    return TestLoader
if __name__ == '__main__':
    TrainLoader, ValLoader = get_data_loader(0, 'zero_vs_one_', 90)
    for matrix, feature, crs in TrainLoader:
        print(matrix.shape, feature.shape, crs.shape)
        break
    print(' ')
    for matrix, feature, crs in ValLoader:
        print(matrix.shape, feature.shape, crs)
        break
