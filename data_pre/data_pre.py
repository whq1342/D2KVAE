import numpy as np
from torch.utils.data import DataLoader
from sklearn import preprocessing
import torch
from data_pre.dataset import MyDataset



def slide_window(data, width, stride):
    sw_data = np.zeros(((int((data.shape[0] - width + stride) / stride), width, data.shape[1])))
    for i in range(sw_data.shape[0]):
        sw_data[i, :, :] = data[i * stride:width + i * stride, :]
    return sw_data

def distance(sequence):
    ones_indices = np.nonzero(sequence)[0]
    output = np.full_like(sequence, -1)
    if len(ones_indices) == 0:
        return output
    last_one_index = ones_indices[-1]
    for i in range(len(sequence)):
        if sequence[i] == 1:
            output[i] = 0
        elif i < last_one_index:
            next_one_index = ones_indices[ones_indices > i][0]
            output[i] = next_one_index - i
    return output
    


def data_pre(data_id, path, label_rate, seq_len, stride, device):
    if data_id =='SRU':
        data = np.load(path)
        zscore2 = preprocessing.StandardScaler().fit(data)
        # zscore1 = preprocessing.MinMaxScaler().fit(data)
        data = zscore2.transform(data)
        data_real = np.copy(data)
        data[np.random.choice(data.shape[0], int(data.shape[0]*(1-label_rate)), replace=False),-1] = 0
        train_data = np.copy(data[:8000, :])
        train_data_real = np.copy(data_real[:8000,:])
        val_data = test_data = np.copy(data[8000:, :])
        val_data_real = test_data_real = np.copy(data_real[8000:,:])
        label_index = np.where(data[:,-1] != 0, 1, 0).reshape(-1, 1)
        label_index = distance(label_index).reshape(-1, 1)
        train_label_index = label_index[:8000, :]
        val_label_index = label_index[8000:, :]
        train_x = slide_window(train_data[:, :-1], width=seq_len, stride=stride)
        train_y = slide_window(train_data[:, -1:], width=seq_len, stride=stride)
        train_index = slide_window(train_label_index, width=seq_len, stride=stride)
        train_y_real = slide_window(train_data_real[:, -1:], width=seq_len, stride=stride)
        val_x = slide_window(val_data[:, :-1], width=seq_len, stride=stride)
        val_y = slide_window(val_data[:, -1:], width=seq_len, stride=stride)
        val_index = slide_window(val_label_index, width=seq_len, stride=stride)
        val_y_real = slide_window(val_data_real[:, -1:], width=seq_len, stride=stride)
        test_x = slide_window(test_data[:, :-1], width=seq_len, stride=stride)
        test_y = slide_window(test_data_real[:, -1:], width=seq_len, stride=stride)
        train_x = np.transpose(train_x, (1, 0, 2))
        train_y = np.transpose(train_y, (1, 0, 2))
        train_index = np.transpose(train_index, (1, 0, 2))
        val_x = np.transpose(val_x, (1, 0, 2))
        val_y = np.transpose(val_y, (1, 0, 2))
        val_index = np.transpose(val_index, (1, 0, 2))
        test_x = np.transpose(test_x, (1, 0, 2))
        train_dataset = MyDataset(torch.tensor(train_x, dtype=torch.float32, device=device),
                                torch.tensor(train_y, dtype=torch.float32, device=device),
                                torch.tensor(train_index, dtype=torch.int, device=device),'3D')
        val_dataset = MyDataset(torch.tensor(val_x, dtype=torch.float32, device=device),
                                torch.tensor(val_y, dtype=torch.float32, device=device),
                                torch.tensor(val_index, dtype=torch.int, device=device),'3D')
    return train_dataset, val_dataset, zscore2, test_x, test_y