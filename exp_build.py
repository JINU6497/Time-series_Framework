import os
import time
import warnings
import numpy as np
import pandas as pd
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from lion_pytorch import Lion

from models import DLinear, TimesNet

from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, check_graph
from utils.utils import progress_bar
from utils.metrics import metric

from data_provider.data_build import data_builder

import matplotlib.pyplot as plt
import pdb

warnings.filterwarnings('ignore')

class Exp_builder():
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

# 모델 사용하려면 여기 추가
    def _build_model(self):
        model_dict = {
            'DLinear': DLinear,
            'TimesNet': TimesNet,
            # 'PatchTST': PatchTST
        }
        model = model_dict[self.args.model].Model(self.args).float()
        return model

    def _get_data(self, type):
        data_set, data_loader = data_builder(self.args, type)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.optim == 'gelu':
            model_optim = F.gelu(self.model.parameters(), lr = self.args.learning_rate)
        elif self.args.optim == 'adamw':
            model_optim = optim.AdamW(self.model.parameters(), lr = self.args.learning_rate)
        elif self.args.optim == 'adam':
            model_optim = optim.Adam(self.model.parameters(), lr = self.args.learning_rate)
        elif self.args.optim == 'SGD':
            model_optim = optim.SGD(self.model.parameters(), lr = self.args.learning_rate)
        elif self.args.optim == 'lion':
            model_optim = Lion(self.model.parameters(), lr = self.args.learning_rate) 
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'mse':
            criterion = nn.MSELoss()
        elif self.args.loss == 'mae':
            criterion = nn.L1Loss()
        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data(type='train')
        
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(type='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        # pdb.set_trace()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_stamp, batch_y_stamp) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_stamp = batch_x_stamp.float().to(self.device)
                batch_y_stamp = batch_y_stamp.float().to(self.device)

                # decoder input
                # Pred_len + label_len 만큼의 Input 만들어짐
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if 'Linear' in self.args.model:
                    outputs = self.model(batch_x)
                elif 'TimesNet' in self.args.model:
                    outputs = self.model(batch_x, batch_x_stamp, dec_inp, batch_y_stamp, batch_y)
                
                # print(outputs.shape,batch_y.shape)
                if self.args.features == 'MS':
                    f_dim = -1
                else:
                    f_dim = 0
                
                # Loss 계산은 pred_len으로만
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            
            if not self.args.train_only:
                vali_loss = self.vali(vali_data, vali_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))
                early_stopping(vali_loss, self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_stamp, batch_y_stamp) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_stamp = batch_x_stamp.float().to(self.device)
                batch_y_stamp = batch_y_stamp.float().to(self.device)

                # decoder input
                # Pred_len + label_len 만큼의 Input 만들어짐
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if 'Linear' in self.args.model:
                    outputs = self.model(batch_x)
                elif 'TimesNet' in self.args.model:
                    outputs = self.model(batch_x, batch_x_stamp, dec_inp, batch_y_stamp, batch_y)
                
                # print(outputs.shape,batch_y.shape)
                if self.args.features == 'MS':
                    f_dim = -1
                else:
                    f_dim = 0
                
                # Loss 계산은 pred_len으로만
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(type='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_stamp, batch_y_stamp) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_stamp = batch_x_stamp.float().to(self.device)
                batch_y_stamp = batch_y_stamp.float().to(self.device)

                # decoder input
                # Pred_len + label_len 만큼의 Input 만들어짐
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if 'Linear' in self.args.model:
                    outputs = self.model(batch_x)
                elif 'TimesNet' in self.args.model:
                    outputs = self.model(batch_x, batch_x_stamp, dec_inp, batch_y_stamp, batch_y)
                
                # print(outputs.shape,batch_y.shape)
                if self.args.features == 'MS':
                    f_dim = -1
                else:
                    f_dim = 0
                
                # Loss 계산은 pred_len으로만
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 50 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.png'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(type='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_stamp, batch_y_stamp) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_stamp = batch_x_stamp.float().to(self.device)
                batch_y_stamp = batch_y_stamp.float().to(self.device)

                # decoder input
                # Pred_len + label_len 만큼의 Input 만들어짐
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if 'Linear' in self.args.model:
                    outputs = self.model(batch_x)
                elif 'TimesNet' in self.args.model:
                    outputs = self.model(batch_x, batch_x_stamp, dec_inp, batch_y_stamp, batch_y)
                
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = np.concatenate(preds, axis=0)
        if (pred_data.scale):
            preds = pred_data.inverse_transform(preds)
        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1), columns=pred_data.cols).to_csv(folder_path + 'real_prediction.csv', index=False)

        return
