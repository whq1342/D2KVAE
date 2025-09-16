import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import pandas as pd
import random
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from torch.utils.data import DataLoader
import time
import argparse
from data_pre.data_pre import data_pre
from model.D2KVAE import D2KVAE
from train.trainer import Trainer

parser = argparse.ArgumentParser(description='sensor')
parser.add_argument('--number', default=0, type=int, help='ID')
parser.add_argument('--epoch', default=300, type=int, help='the number of epochs')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--batch_size', default=256, type=int, help='batch_size')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument("--gpu", type=int, default=0, help="GPU编号,-1表示不使用GPU")
parser.add_argument('--z_x_dim', default=12, type=int, help='z_x_dim')
parser.add_argument('--z_y_dim', default=20, type=int, help='z_y_dim')
parser.add_argument('--dense_x_g', nargs='+', default=[12], type=int, help='dense_x_g')# x_t to g_t
parser.add_argument('--dense_h_x', nargs='+', default=[8], type=int, help='dense_h_x')# h_t to x_t
parser.add_argument('--dense_z_y', nargs='+', default=[25], type=int, help='dense_z_y')# z_x_t & z_y_tm1 to z_y_t
parser.add_argument('--dense_h_y', nargs='+', default=[64], type=int, help='dense_h_y')# h_y_t to y_t
parser.add_argument('--dense_gz_z', default=[], type=list, help='dense_gz_z')# g_t & z_x_tm1 -> z_x_t
parser.add_argument('--dense_z_h', default=[], type=list, help='dense_z_h')# z_t to h_t
parser.add_argument('--dense_hz_z', default=[], type=list, help='dense_hz_z')# z_x_tm1 -> z_x_t
parser.add_argument('--dense_hz_z_y', default=[], type=list, help='dense_hz_z_y')# z_y_tm1 -> z_y_t
parser.add_argument('--dim_rnn_g', default=128, type=int, help='dim_rnn_g')
parser.add_argument('--dim_rnn_h', default=128, type=int, help='dim_rnn_h')
parser.add_argument('--dim_rnn_y', default=256, type=int, help='dim_rnn_y')
parser.add_argument('--seq_len', default=30, type=int, help='seq_len')
parser.add_argument('--stride', default=5, type=int, help='stride')
parser.add_argument('--label_rate', default=0.5, type=float, help='label_rate')
parser.add_argument('--label_weight', default=2., type=float, help='label_weight')
parser.add_argument('--kl_x_weight', default=0.001, type=float, help='kl_x_weight')
parser.add_argument('--kl_y_weight', default=1., type=float, help='kl_y_weight')
parser.add_argument('--koopman_weight', default=0.01, type=float, help='koopman_weight')
args = parser.parse_args()

seed = args.seed
batch_size = args.batch_size
seq_len = args.seq_len
stride = args.stride
label_rate = args.label_rate
device = torch.device("cuda:" + str(args.gpu))

print('Random Seed:', seed)
print('Label Rate:', label_rate)

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
setup_seed(seed)

train_dataset, val_dataset, zscore2, test_x, test_y = data_pre(data_id='SRU', path = './dataset/SRU_data.npy', label_rate=label_rate, seq_len=seq_len, stride=stride, device=device)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}

custom_model = D2KVAE(x_dim= 20, z_x_dim=args.z_x_dim, z_y_dim=args.z_y_dim, activation='tanh',
                    dense_x_g=args.dense_x_g, dense_gz_z=args.dense_gz_z,  ### inference z
                    dim_RNN_g=args.dim_rnn_g, num_RNN_g=1,  ### inference z
                    dense_z_h=args.dense_z_h, dense_h_x=args.dense_h_x,  ### generation x
                    dim_RNN_h=args.dim_rnn_h, num_RNN_h=1,  ### generation x
                    dense_hz_z=args.dense_hz_z,  ### prior
                    dense_z_y=args.dense_z_y, dense_h_y=args.dense_h_y,  ### regression y
                    dim_RNN_y=args.dim_rnn_y, num_RNN_y=1,  ### regression y
                    dense_hz_z_y=args.dense_hz_z_y,
                    dropout_p=0, beta=1, device=device)

custom_trainer = Trainer(dataloader_dict=dataloader_dict, model=custom_model, epoch=args.epoch, lr=args.lr,
                         scheduler=True, verbose=True, betaSchedule=[4, 1], device=device, zscore=zscore2, args=args)

start_train_time = time.time()
error_index = custom_trainer.train()
end_train_time = time.time()

start_test_time = time.time()
with torch.no_grad():
    outputs = custom_model(torch.tensor(test_x, dtype=torch.float32, device=device))[1]
end_test_time = time.time()
outputs = outputs.cpu().detach().numpy()

outputs = outputs.transpose(1, 0, 2)
test_y_pred = np.concatenate((outputs[0, :, 0], outputs[1:, -stride:, 0].reshape(-1)))
test_y_real = np.concatenate((test_y[0, :, 0], test_y[1:, -stride:, 0].reshape(-1)))

test_y_pred = test_y_pred * zscore2.scale_[-1] + zscore2.mean_[-1]
test_y_real = test_y_real * zscore2.scale_[-1] + zscore2.mean_[-1]
test_rmse, test_mae, test_r2 = np.sqrt(mean_squared_error(test_y_pred, test_y_real)), mean_absolute_error(test_y_pred, test_y_real), r2_score(test_y_pred, test_y_real)

print('test_rmse = ' + str(round(test_rmse*100, 5)))
print('test_mae = ' + str(round(test_mae*100, 5)))
print('r2 = ', str(round(test_r2*100, 5)))
print("Training Time:" + str(round((end_train_time - start_train_time), 5)) + "Seconds")
print("Testing Time:" + str(round((end_test_time - start_test_time), 5)) + "Seconds")

