import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pdb

warnings.filterwarnings('ignore')

# type: train, test, val

class Dataset_custom(Dataset):
    def __init__(self, 
                 root_path: str, 
                 data_path: str, 
                 len_info: list,
                 type: str = 'train', 
                 data_type: str = 'Forecasting',
                 features: str = 'M',
                 target: str = 'target',
                 scaler: str = None,
                 train_only: bool = False,
                 train_ratio: int = 0.7,
                 test_ratio: int = 0.2,
                 time_encode: int = 0, 
                 freq: str = 'h'):
        
        self.data_type = data_type
        self.root_path = root_path
        self.data_path = data_path

        self.window_len = len_info[0]
        self.label_len = len_info[1]
        self.pred_len = len_info[2]

        self.feature = features
        self.target = target
        self.scaler = scaler
        self.time_encode = time_encode
        self.freq = freq

        self.type = type

        self.train_only = train_only
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio 
        self.__read_data__()
        
    def __read_data__(self):
        
        if self.scaler is not None:
            if self.scaler == 'minmax':
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
            elif self.scaler == 'standard':
                self.scaler = StandardScaler()
                
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        if self.data_type == 'Forecasting':
            df_data = df_raw[df_raw.columns[1:]]
            df_date = df_raw[['date']]
            
            if not self.train_only:
                num_train = int(len(df_raw) * (self.train_ratio))
                num_test = int(len(df_raw) * (self.test_ratio))
                num_vali = len(df_raw) - num_train - num_test
                
                train_line = [0, num_train]
                val_line = [num_train - self.window_len, num_train + num_vali]
                test_line = [len(df_raw) - num_test - self.window_len, len(df_raw)]
                
            else:
                num_train = int(len(df_raw) * 1)
                train_line = [0, len(df_raw)]
                
            if self.scaler is not None:
                train_data = df_data[train_line[0]:train_line[1]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values
            
            if self.type == 'train':
                self.data_x = data[train_line[0]:train_line[1]]
                self.data_y = data[train_line[0]:train_line[1]]
                df_stamp = df_raw[['date']][train_line[0]:train_line[1]]
                df_stamp['date'] = pd.to_datetime(df_stamp.date)
                
                if self.time_encode == 0:
                    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                    df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                    data_stamp = df_stamp.drop(['date'], axis = 1).values
                    self.data_stamp = data_stamp
                    
            elif self.type == 'val':
                self.data_x = data[val_line[0]:val_line[1]]
                self.data_y = data[val_line[0]:val_line[1]]
                df_stamp = df_raw[['date']][val_line[0]:val_line[1]]
                df_stamp['date'] = pd.to_datetime(df_stamp.date)
                if self.time_encode == 0:
                    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                    df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                    data_stamp = df_stamp.drop(['date'], axis = 1).values
                    self.data_stamp = data_stamp
                    
            elif self.type == 'test':
                self.data_x = data[test_line[0]:test_line[1]]
                self.data_y = data[test_line[0]:test_line[1]]
                df_stamp = df_raw[['date']][test_line[0]:test_line[1]]
                df_stamp['date'] = pd.to_datetime(df_stamp.date)
                if self.time_encode == 0:
                    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                    df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                    data_stamp = df_stamp.drop(['date'], axis = 1).values
                    self.data_stamp = data_stamp
                
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.window_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin : s_end]
        seq_y = self.data_y[r_begin : r_end]
        seq_x_stamp = self.data_stamp[s_begin:s_end]
        seq_y_stamp = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_stamp, seq_y_stamp
    
    def __len__(self):
        return len(self.data_x) - self.window_len - self.pred_len + 1

    def inverse_transform(self, data):
        if self.scaler == 'minmax':
            self.scaler.inverse_transform(data)
        elif self.scaler == 'standard':
            self.scaler.inverse_transform(data)
    
    
    
    
    
    

def load_dataset(dataname: str,
                 datainfo: dict,
                 subdataname:str = None,
                 val_rate: float = 0.8):
    
    if dataname == 'HAMON':
        train = pd.read_csv(os.path.join(datainfo.data_dir, f'Train_{subdataname}.csv'), index_col=0).interpolate()
        train_timestamp = train.index
        valid_split_index = int(len(train) * val_rate)
        valid = train.iloc[valid_split_index:].to_numpy()
        train = train.iloc[:valid_split_index].to_numpy()
        valid_timestamp = train_timestamp[valid_split_index:].to_numpy()
        train_timestamp = train_timestamp[:valid_split_index].to_numpy()
        test = pd.read_csv(os.path.join(datainfo.data_dir, f'Test_{subdataname}.csv'), index_col=0).interpolate()
        test_timestamp = test.index
        test_label = pd.read_csv(os.path.join(datainfo.data_dir, f'Test_Label_{subdataname}.csv'), index_col=0).to_numpy()
        
    if dataname == 'PSM':
        train = pd.read_csv(os.path.join(datainfo.data_dir, f'Train_{subdataname}.csv'), index_col=0).interpolate()
        train_timestamp = train.index
        valid_split_index = int(len(train) * val_rate)
        valid = train.iloc[valid_split_index:].to_numpy()
        train = train.iloc[:valid_split_index].to_numpy()
        valid_timestamp = train_timestamp[valid_split_index:].to_numpy()
        train_timestamp = train_timestamp[:valid_split_index].to_numpy()
        test = pd.read_csv(os.path.join(datainfo.data_dir, f'Test_{subdataname}.csv'), index_col=0).interpolate()
        test_timestamp = test.index
        test_label = pd.read_csv(os.path.join(datainfo.data_dir, f'Test_Label_{subdataname}.csv'), index_col=0).to_numpy()
    
    return train, train_timestamp, valid, valid_timestamp, test, test_timestamp, test_label


