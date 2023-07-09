import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
import torch.optim as optim

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""1. Convolution Encoder"""

class convEncoder(nn.Module):
    def __init__(self):
        super(convEncoder, self).__init__()
        
        self.Conv1 = nn.Conv3d(in_channels = 3, out_channels= 32, kernel_size = (1, 3, 3), stride = (1, 1, 1), padding =(0,1,1))
        self.Conv2 = nn.Conv3d(in_channels = 32, out_channels= 64, kernel_size = (1, 3, 3), stride = (1, 2, 2), padding =(0,1,1))
        self.Conv3 = nn.Conv3d(in_channels = 64, out_channels= 128, kernel_size = (1, 2, 2), stride = (1, 2, 2), padding =(0,1,1))
        self.Conv4 = nn.Conv3d(in_channels = 128, out_channels= 256, kernel_size = (1, 2, 2), stride = (1, 2, 2))
        
    def forward(self, input):
        
        """input shape = (batch, timestep, feature, faeture, lag(Channel))"""
        
        encoder1_out = F.selu(self.Conv1(input))
        encoder2_out = F.selu(self.Conv2(encoder1_out))
        encoder3_out = F.selu(self.Conv3(encoder2_out))
        encoder4_out = F.selu(self.Conv4(encoder3_out))
        return encoder1_out, encoder2_out, encoder3_out, encoder4_out
    

"""2. Attention based ConvLSTM"""

class ConvLSTM_cell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super (ConvLSTM_cell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        
        # Input gate 
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        # Forget gate
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        # Cell gate
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        # Output gate
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        # 처음엔 이전 정보로부터 받을 정보가 없어서 None, 이후 learnable한 parameter로 Initialization.
        self.Wci = None
        self.Wcf = None
        self.Wco = None 
        
    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        # Hidden state, Cell state
        return ch, cc
    
    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())
        
class ConvLSTM(nn.Module):

    def __init__(self, in_channels=32, h_channels=[32], kernel_size=3, seq_len=1, attention=True):

        super(ConvLSTM, self).__init__()
        self.input_channels = [in_channels] + h_channels
        self.hidden_channels = h_channels
        self.attention = attention
        self.kernel_size = kernel_size
        self.num_layers = len(h_channels)
        self.seq_len = seq_len
        self._all_layers = []
        self.flatten = nn.Flatten()
        self.alpha_i = None
        for i in range(self.num_layers):
            cell = ConvLSTM_cell(self.input_channels[i], self.hidden_channels[i], self.kernel_size).cuda()
            self._all_layers.append(cell)

    def forward(self, input):

        """
        input with shape: (batch, seq_len, num_channels, height, width)
        """
        
        internal_state = []
        outputs = []
        # 각 Timestep 마다 
        for timestep in range(self.seq_len):
            x = input[:, timestep, ...]
            for i in range(self.num_layers):
                # Timestep == 0 인 경우는 Initialization
                if timestep == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = self._all_layers[i].init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))
                # Hidden state, Cell state
                (h, c) = internal_state[i]
                
                # 다음 시점의 Hidden state, Cell state
                x, new_c = self._all_layers[i](x, h, c)
                
                internal_state[i] = (x, new_c)

            outputs.append(x)
        outputs = torch.stack(outputs).permute(1, 0, 2, 3, 4)

        if self.attention:
            # wegiht 구하는 과정
            alpha_i = [torch.einsum('ij,kj->i', self.flatten(outputs[:, i, ...]), self.flatten(outputs[:, -1, ...]))
                       for i in range(self.seq_len)]
            alpha_i = torch.stack(alpha_i, dim=1) / self.seq_len
            alpha_i = F.softmax(alpha_i, dim=1)
            self.alpha_i = alpha_i
            # Weight가 적용된 Output 값 도출 
            x = torch.einsum('ijklm, ij -> iklm', outputs, alpha_i)
        return outputs, (x, new_c)
    

def calculate_score(true, pred):
    residual_matrix = true - pred
    err = np.sum(np.sum(np.sum(residual_matrix**2, axis=1), axis=1), axis = 1)
    return err, residual_matrix

def search_attack(attack, timestep):
    del attack[0:timestep-1]
    return attack    

""""3. Model"""

class Model(nn.Module):

    def __init__(self, num_timesteps=5, attention=True):
        super().__init__()
        
        self.cnn_encoder = convEncoder()
        
        self.ConvLSTM1 = ConvLSTM(in_channels=32, h_channels=[32], kernel_size=3, seq_len=num_timesteps,
                                  attention=attention)
        self.ConvLSTM2 = ConvLSTM(in_channels=64, h_channels=[64], kernel_size=3, seq_len=num_timesteps,
                                  attention=attention)
        self.ConvLSTM3 = ConvLSTM(in_channels=128, h_channels=[128], kernel_size=3, seq_len=num_timesteps,
                                  attention=attention)
        self.ConvLSTM4 = ConvLSTM(in_channels=256, h_channels=[256], kernel_size=3, seq_len=num_timesteps,
                                  attention=attention)
        
        self.Deconv4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.Deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=2, stride=2, padding=1)
        self.Deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.Deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        """
        input X with shape: (batch, seq_len, num_channels, height, width)
        """
        
        conv1_out, conv2_out, conv3_out, conv4_out = self.cnn_encoder(x)
        
        # ConvLSTM을 통하여 나오는 Output은 Weight가 적용된 output값

        x_c1_seq = conv1_out
        _, (x_c1, _) = self.ConvLSTM1(x_c1_seq.permute(0, 2, 1, 3, 4))

        x_c2_seq = conv2_out
        _, (x_c2, _) = self.ConvLSTM2(x_c2_seq.permute(0, 2, 1, 3, 4))

        x_c3_seq = conv3_out
        _, (x_c3, _) = self.ConvLSTM3(x_c3_seq.permute(0, 2, 1, 3, 4))

        x_c4_seq = conv4_out
        _, (x_c4, _) = self.ConvLSTM4(x_c4_seq.permute(0, 2, 1, 3, 4))

        x_d4 = F.selu(self.Deconv4.forward(x_c4, output_size=[x_c3.shape[-1], x_c3.shape[-2]]))

        x_d3 = torch.cat((x_d4, x_c3), dim=1)
        x_d3 = F.selu(self.Deconv3.forward(x_d3, output_size=[x_c2.shape[-1], x_c2.shape[-2]]))

        x_d2 = torch.cat((x_d3, x_c2), dim=1)
        x_d2 = F.selu(self.Deconv2.forward(x_d2, output_size=[x_c1.shape[-1], x_c1.shape[-2]]))

        x_d1 = torch.cat((x_d2, x_c1), dim=1)
        x_rec = F.selu(self.Deconv1.forward(x_d1, output_size=[x_c1.shape[-1], x_c1.shape[-2]]))

        return x_rec