import torch
from torch import nn
import torch.nn.functional as F

import math
import copy
from datetime import datetime
import time


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class EncoderDecoder(nn.Module):
    def __init__(self, input_len, hidden_units, num_layers, activate_func, batch_first=True):
        # Ensure hidden_units and num_layers are lists
        assert isinstance(hidden_units, list), "hidden_units must be a list"
        assert isinstance(num_layers, list), "num_layers must be a list"
        assert len(hidden_units) == len(num_layers), "hidden_units and num_layers must have the same length"

        super(EncoderDecoder, self).__init__()
        self.input_len = input_len
        self.hidden_size = hidden_units
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.activate_func = activate_func
        self.encoder = nn.LSTM(input_size=input_len, hidden_size=hidden_units[0], num_layers=num_layers[0],
                               batch_first=True)
        self.lstms = nn.ModuleList()
        for i in range(1, len(hidden_units)):
            self.lstms.append(
                nn.LSTM(input_size=hidden_units[i - 1], hidden_size=hidden_units[i], num_layers=num_layers[i],
                        batch_first=True))

    def forward(self, x):
        modified_x = self.activate_func(self.encoder(x)[0])
        for i in range(len(self.hidden_size) - 1):
            modified_x = self.activate_func(self.lstms[i](modified_x)[0])
        return modified_x



class MultiLayerDNN(nn.Module):
    def __init__(self, input_len, hidden_units, activate_func, batch_first=True):
        # Ensure hidden_units and num_layers are lists
        assert isinstance(hidden_units, list), "hidden_units must be a list"

        super(MultiLayerDNN, self).__init__()
        self.input_len = input_len
        self.hidden_size = hidden_units
        self.batch_first = batch_first
        self.activate_func = activate_func
        self.encoder = nn.Linear(in_features=input_len, out_features=hidden_units[0], bias=True)
        self.dnns = nn.ModuleList()
        for i in range(1, len(hidden_units)):
            self.dnns.append(
                nn.Linear(in_features=hidden_units[i - 1], out_features=hidden_units[i], bias=True))

    def forward(self, x):
        modified_x = self.activate_func(self.encoder(x))
        for i in range(len(self.hidden_size) - 1):
            modified_x = self.activate_func(self.dnns[i](modified_x))
        return modified_x


class FixedAttention(nn.Module):
    def __init__(self, attention):
        super(FixedAttention, self).__init__()
        self.attention = attention

    def forward(self, x):
        for i in range(len(x)):
            for j in range(int(len(x[i]) / 6)):
                x[i][6 * j + 3] = x[i][6 * j + 3] * self.attention
                x[i][6 * j + 4] = x[i][6 * j + 4] * self.attention
        return x


class ForwardFixAttentionLSTM(nn.Module):
    def __init__(self, attention, input_len, hidden_units, out_len, num_layers, activate_func):
        super(ForwardFixAttentionLSTM, self).__init__()

        # Ensure hidden_units and num_layers are lists
        assert isinstance(hidden_units, list), "hidden_units must be a list"
        assert isinstance(num_layers, list), "num_layers must be a list"
        assert len(hidden_units) == len(num_layers), "hidden_units and num_layers must have the same length"

        # Parameters
        self.attention = attention
        self.input_len = input_len
        self.hidden_size = hidden_units
        self.num_layers = num_layers
        self.hidden = None
        # self.num_lstms = num_lstms
        self.out_len = out_len
        self.activate_func = activate_func

        # Layers
        self.fixed_attention = FixedAttention(attention)
        '''
        self.encoder = nn.LSTM(input_size=input_len, hidden_size=hidden_units[0], num_layers=num_layers[0],
                               batch_first=True)
        self.lstms = nn.ModuleList()
        for i in range(1, len(hidden_units)):
            self.lstms.append(
                nn.LSTM(input_size=hidden_units[i - 1], hidden_size=hidden_units[i], num_layers=num_layers[i],
                        batch_first=True))
        # self.lstms = get_clones(
        #    nn.LSTM(input_size=hidden_units, hidden_size=hidden_units, num_layers=num_layers, batch_first=True),
        #    num_lstms)
        '''
        self.encoder_decoder = EncoderDecoder(input_len=self.input_len, hidden_units=self.hidden_size,
                                              num_layers=self.num_layers, activate_func=self.activate_func,
                                              batch_first=True)
        self.feedforward = nn.Linear(in_features=hidden_units[-1], out_features=hidden_units[-1], bias=True)
        self.fc1 = nn.Linear(in_features=hidden_units[-1], out_features=out_len, bias=True)

    def forward(self, x):
        attentioned_x = self.fixed_attention(x)
        '''
        modified_x = F.relu(self.encoder(attentioned_x)[0])
        for i in range(len(self.hidden_size) - 1):
            modified_x = F.relu(self.lstms[i](modified_x)[0])
        '''
        modified_x = self.encoder_decoder(attentioned_x)
        # modified_x = F.relu(modified_x + self.feedforward(modified_x))  # residual
        out = self.fc1(modified_x)
        out = torch.sigmoid(out)
        return out, self.hidden


class SelfAttention(nn.Module):
    def __init__(self, input_len, out_len):
        super(SelfAttention, self).__init__()
        self.input_len = input_len
        self.out_len = out_len
        self.attention = nn.Linear(self.input_len, self.out_len, bias=False)

    def forward(self, x):
        attentioned_x = self.attention(x)
        return F.relu(attentioned_x)


class ForwardSelfAttentionLSTM(nn.Module):
    def __init__(self, input_len, hidden_units, out_len, num_layers, activate_func):
        super(ForwardSelfAttentionLSTM, self).__init__()

        # Ensure hidden_units and num_layers are lists
        assert isinstance(hidden_units, list), "hidden_units must be a list"
        assert isinstance(num_layers, list), "num_layers must be a list"
        assert len(hidden_units) == len(num_layers), "hidden_units and num_layers must have the same length"

        # Parameters
        self.input_len = input_len
        self.hidden_size = hidden_units
        self.num_layers = num_layers
        self.hidden = None
        # self.num_lstms = num_lstms
        self.out_len = out_len
        self.activate_func = activate_func

        # Layers
        self.self_attention = nn.Linear(in_features=self.input_len, out_features=self.input_len,
                                        bias=False)  # Self attention layer
        '''
        self.encoder = nn.LSTM(input_size=self.input_len, hidden_size=hidden_units[0],
                               num_layers=num_layers[0], batch_first=True)
        self.lstms = nn.ModuleList()
        for i in range(1, len(hidden_units)):
            self.lstms.append(
                nn.LSTM(input_size=hidden_units[i - 1], hidden_size=hidden_units[i], num_layers=num_layers[i],
                        batch_first=True))
        # self.lstms = get_clones(
        #    nn.LSTM(input_size=hidden_units, hidden_size=hidden_units, num_layers=num_layers, batch_first=True),
        #    num_lstms)
        '''
        self.encoder_decoder = EncoderDecoder(input_len=self.input_len, hidden_units=self.hidden_size,
                                              num_layers=self.num_layers, activate_func=self.activate_func,
                                              batch_first=True)
        self.feedforward = nn.Linear(in_features=hidden_units[-1], out_features=hidden_units[-1], bias=True)
        self.fc1 = nn.Linear(in_features=hidden_units[-1], out_features=out_len, bias=True)

    def forward(self, x):
        attentioned_x = self.self_attention(x)
        '''
        modified_x = F.relu(self.encoder(attentioned_x)[0])
        for i in range(len(self.hidden_size) - 1):
            modified_x = F.relu(self.lstms[i](modified_x)[0])
        '''
        modified_x = self.encoder_decoder(attentioned_x)
        # modified_x = F.relu(modified_x + self.feedforward(modified_x))  # residual
        out = self.fc1(modified_x)
        out = torch.sigmoid(out)
        return out, self.hidden


class ForwardSelfCDLSTM(nn.Module):
    def __init__(self, input_len, hidden_units, out_len, num_layers, activate_func):
        super(ForwardSelfCDLSTM, self).__init__()

        # Ensure hidden_units and num_layers are lists
        assert isinstance(hidden_units, list), "hidden_units must be a list"
        assert isinstance(num_layers, list), "num_layers must be a list"
        assert len(hidden_units) == len(num_layers), "hidden_units and num_layers must have the same length"

        # Parameters
        self.input_len = input_len
        self.hidden_size = hidden_units
        self.num_layers = num_layers
        self.hidden = None
        # self.num_lstms = num_lstms
        self.out_len = out_len
        self.activate_func = activate_func

        # Layers
        self.self_attention = nn.Linear(in_features=self.input_len, out_features=self.input_len,
                                        bias=False)  # Self attention layer
        '''
        self.encoder = nn.LSTM(input_size=self.input_len, hidden_size=hidden_units[0],
                               num_layers=num_layers[0], batch_first=True)
        self.lstms = nn.ModuleList()
        for i in range(1, len(hidden_units)):
            self.lstms.append(
                nn.LSTM(input_size=hidden_units[i - 1], hidden_size=hidden_units[i], num_layers=num_layers[i],
                        batch_first=True))
        # self.lstms = get_clones(
        #    nn.LSTM(input_size=hidden_units, hidden_size=hidden_units, num_layers=num_layers, batch_first=True),
        #    num_lstms)
        '''
        self.encoder_decoder = EncoderDecoder(input_len=self.input_len, hidden_units=self.hidden_size,
                                              num_layers=self.num_layers, activate_func=self.activate_func,
                                              batch_first=True)
        self.feedforward = nn.Linear(in_features=hidden_units[-1], out_features=hidden_units[-1], bias=True)
        self.fc1 = nn.Linear(in_features=hidden_units[-1], out_features=out_len, bias=True)

    def forward(self, x):
        attentioned_x = self.self_attention(x)
        '''
        modified_x = F.relu(self.encoder(attentioned_x)[0])
        for i in range(len(self.hidden_size) - 1):
            modified_x = F.relu(self.lstms[i](modified_x)[0])
        '''
        modified_x = self.encoder_decoder(attentioned_x)
        # modified_x = F.relu(modified_x + self.feedforward(modified_x))  # residual
        out = self.fc1(modified_x)
        out = 10 * torch.tanh(out)
        return out, self.hidden


class SelfAttentionKQV(nn.Module):
    def __init__(self, input_len):
        super(SelfAttentionKQV, self).__init__()
        self.input_len = input_len
        self.q_linear = nn.Linear(input_len, input_len)
        self.k_linear = nn.Linear(input_len, input_len)
        self.v_linear = nn.Linear(input_len, input_len)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Generate Q, K, V
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.input_len ** 0.5)
        attn_weights = self.softmax(attn_scores)

        # Compute weighted sum
        # output = torch.matmul(attn_weights, V)
        output = torch.matmul(attn_weights, x)
        return output, attn_weights


class ForwardSelfAttentionKQVLSTM(nn.Module):
    def __init__(self, input_len, hidden_units, out_len, num_layers):
        super(ForwardSelfAttentionKQVLSTM, self).__init__()

        # Ensure hidden_units and num_layers are lists
        assert isinstance(hidden_units, list), "hidden_units must be a list"
        assert isinstance(num_layers, list), "num_layers must be a list"
        assert len(hidden_units) == len(num_layers), "hidden_units and num_layers must have the same length"

        # Parameters
        self.input_len = input_len
        self.hidden_size = hidden_units
        self.num_layers = num_layers
        self.hidden = None
        # self.num_lstms = num_lstms
        self.out_len = out_len

        # Layers
        self.self_attention = SelfAttentionKQV(input_len=self.input_len)
        self.encoder = nn.LSTM(input_size=self.input_len, hidden_size=hidden_units[0],
                               num_layers=num_layers[0], batch_first=True)
        self.lstms = nn.ModuleList()
        for i in range(1, len(hidden_units)):
            self.lstms.append(
                nn.LSTM(input_size=hidden_units[i - 1], hidden_size=hidden_units[i], num_layers=num_layers[i],
                        batch_first=True))
        # self.lstms = get_clones(
        #    nn.LSTM(input_size=hidden_units, hidden_size=hidden_units, num_layers=num_layers, batch_first=True),
        #    num_lstms)
        self.feedforward = nn.Linear(in_features=hidden_units[-1], out_features=hidden_units[-1], bias=True)
        self.fc1 = nn.Linear(in_features=hidden_units[-1], out_features=out_len, bias=True)

    def forward(self, x):
        attn_output, attn_weights = self.self_attention(x)
        modified_x = F.relu(self.encoder(attn_output)[0])
        for i in range(len(self.hidden_size) - 1):
            modified_x = F.relu(self.lstms[i](modified_x)[0])

        # modified_x = F.relu(modified_x + self.feedforward(modified_x))  # residual
        out = self.fc1(modified_x)
        out = torch.sigmoid(out)
        return out, self.hidden


class ForwardMultiheadAttentionLSTM(nn.Module):
    def __init__(self, input_len, hidden_units, out_len, num_layers, num_heads=1):
        super(ForwardMultiheadAttentionLSTM, self).__init__()

        # Ensure hidden_units and num_layers are lists
        assert isinstance(hidden_units, list), "hidden_units must be a list"
        assert isinstance(num_layers, list), "num_layers must be a list"
        assert len(hidden_units) == len(num_layers), "hidden_units and num_layers must have the same length"

        # Parameters
        self.input_len = input_len
        self.hidden_size = hidden_units
        self.num_layers = num_layers
        self.hidden = None
        # self.num_lstms = num_lstms
        self.out_len = out_len
        self.num_heads = num_heads
        self.query = None
        self.key = None
        self.value = None

        # Layers
        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.input_len,
                                                         num_heads=self.num_heads, batch_first=True)
        self.encoder = nn.LSTM(input_size=input_len, hidden_size=hidden_units[0], num_layers=num_layers[0],
                               batch_first=True)
        self.lstms = nn.ModuleList()
        for i in range(1, len(hidden_units)):
            self.lstms.append(
                nn.LSTM(input_size=hidden_units[i - 1], hidden_size=hidden_units[i], num_layers=num_layers[i],
                        batch_first=True))
        # self.lstms = get_clones(
        #    nn.LSTM(input_size=hidden_units, hidden_size=hidden_units, num_layers=num_layers, batch_first=True),
        #    num_lstms)
        # self.multihead_attention = nn.MultiheadAttention(embed_dim=self.hidden_size[-1],
        #                                                  num_heads=self.num_heads, batch_first=True)
        self.feedforward = nn.Linear(in_features=hidden_units[-1], out_features=hidden_units[-1], bias=True)
        self.fc1 = nn.Linear(in_features=hidden_units[-1], out_features=out_len, bias=True)

    def forward(self, x):
        attentioned_x, _ = self.multihead_attention(x, x, x)
        modified_x = F.relu(self.encoder(attentioned_x)[0])
        for i in range(len(self.hidden_size) - 1):
            modified_x = F.relu(self.lstms[i](modified_x)[0])
        # modified_x = F.relu(modified_x + self.feedforward(modified_x))  # residual
        # attentioned_x, _ = self.multihead_attention(modified_x, modified_x, modified_x)
        # attentioned_x shape: (batch_size, seq_length, hidden_dim)
        # Selecting the output corresponding to the last sequence element
        out = self.fc1(modified_x)
        # out = self.fc1(modified_x)
        out = torch.sigmoid(out)
        return out, self.hidden


class VectorAttention(nn.Module):
    def __init__(self, input_size):
        super(VectorAttention, self).__init__()
        self.input_size = input_size
        self.attention_vector = nn.Parameter(torch.randn(input_size))
        self.attention_vector.requires_grad = True

    def forward(self, x):
        dot_x = x * self.attention_vector
        return dot_x


class ForwardVectorAttentionLSTM(nn.Module):
    def __init__(self, input_len, hidden_units, out_len, num_layers, activate_func):
        super(ForwardVectorAttentionLSTM, self).__init__()

        # Ensure hidden_units and num_layers are lists
        assert isinstance(hidden_units, list), "hidden_units must be a list"
        assert isinstance(num_layers, list), "num_layers must be a list"
        assert len(hidden_units) == len(num_layers), "hidden_units and num_layers must have the same length"

        # Parameters
        self.input_len = input_len
        self.hidden_size = hidden_units
        self.num_layers = num_layers
        self.hidden = None
        self.out_len = out_len
        self.activate_func = activate_func

        # Layers
        self.vector_attention = VectorAttention(self.input_len)
        self.encoder_decoder = EncoderDecoder(input_len=self.input_len, hidden_units=self.hidden_size,
                                              num_layers=self.num_layers, activate_func=self.activate_func,
                                              batch_first=True)
        self.feedforward = nn.Linear(in_features=hidden_units[-1], out_features=hidden_units[-1], bias=True)
        self.fc1 = nn.Linear(in_features=hidden_units[-1], out_features=out_len, bias=True)

    def forward(self, x):
        attentioned_x = self.vector_attention(x)
        modified_x = self.encoder_decoder(attentioned_x)
        # modified_x = F.relu(modified_x + self.feedforward(modified_x))  # residual
        out = self.fc1(modified_x)
        out = torch.sigmoid(out)
        return out, self.hidden


class ForwardSelfAttentionDNN(nn.Module):
    def __init__(self, input_len, hidden_units, out_len, activate_func):
        super(ForwardSelfAttentionDNN, self).__init__()

        # Ensure hidden_units and num_layers are lists
        assert isinstance(hidden_units, list), "hidden_units must be a list"

        # Parameters
        self.input_len = input_len
        self.hidden_size = hidden_units
        self.hidden = None
        self.out_len = out_len
        self.activate_func = activate_func

        # Layers
        self.self_attention = nn.Linear(in_features=self.input_len, out_features=self.input_len,
                                        bias=False)  # Self attention layer
        self.dnns = MultiLayerDNN(input_len=self.input_len, hidden_units=self.hidden_size,
                                  activate_func=self.activate_func, batch_first=True)
        self.feedforward = nn.Linear(in_features=hidden_units[-1], out_features=hidden_units[-1], bias=True)
        self.fc1 = nn.Linear(in_features=hidden_units[-1], out_features=out_len, bias=True)

    def forward(self, x):
        attentioned_x = self.self_attention(x)
        modified_x = self.dnns(attentioned_x)
        # modified_x = F.relu(modified_x + self.feedforward(modified_x))  # residual
        out = self.fc1(modified_x)
        out = torch.sigmoid(out)
        return out, self.hidden



class ForwardNoAttentionDNN(nn.Module):
    def __init__(self, input_len, hidden_units, out_len, activate_func):
        super(ForwardNoAttentionDNN, self).__init__()

        # Ensure hidden_units and num_layers are lists
        assert isinstance(hidden_units, list), "hidden_units must be a list"

        # Parameters
        self.input_len = input_len
        self.hidden_size = hidden_units
        self.hidden = None
        self.out_len = out_len
        self.activate_func = activate_func

        # Layers
        self.dnns = MultiLayerDNN(input_len=self.input_len, hidden_units=self.hidden_size,
                                  activate_func=self.activate_func, batch_first=True)
        self.feedforward = nn.Linear(in_features=hidden_units[-1], out_features=hidden_units[-1], bias=True)
        self.fc1 = nn.Linear(in_features=hidden_units[-1], out_features=out_len, bias=True)

    def forward(self, x):
        modified_x = self.dnns(x)
        # modified_x = F.relu(modified_x + self.feedforward(modified_x))  # residual
        out = self.fc1(modified_x)
        out = torch.sigmoid(out)
        return out, self.hidden


class ForwardVectorAttentionDNN(nn.Module):
    def __init__(self, input_len, hidden_units, out_len, activate_func):
        super(ForwardVectorAttentionDNN, self).__init__()

        # Ensure hidden_units and num_layers are lists
        assert isinstance(hidden_units, list), "hidden_units must be a list"

        # Parameters
        self.input_len = input_len
        self.hidden_size = hidden_units
        self.hidden = None
        self.out_len = out_len
        self.activate_func = activate_func

        # Layers
        self.vector_attention = VectorAttention(self.input_len)
        self.dnns = MultiLayerDNN(input_len=self.input_len, hidden_units=self.hidden_size,
                                  activate_func=self.activate_func, batch_first=True)
        self.feedforward = nn.Linear(in_features=hidden_units[-1], out_features=hidden_units[-1], bias=True)
        self.fc1 = nn.Linear(in_features=hidden_units[-1], out_features=out_len, bias=True)

    def forward(self, x):
        attentioned_x = self.vector_attention(x)
        modified_x = self.dnns(attentioned_x)
        # modified_x = F.relu(modified_x + self.feedforward(modified_x))  # residual
        out = self.fc1(modified_x)
        out = torch.sigmoid(out)
        return out, self.hidden


# Clamp the output within a reasonable range
class Clamp(nn.Module):
    def __init__(self, x_mean, y_mean, z_mean, l_mean, t_mean, x_std, y_std, z_std, l_std, t_std, device):
        super(Clamp, self).__init__()
        w_mean, w_std = [36.66], [24.55]
        self.x_mean = x_mean[0]
        self.y_mean = y_mean[0]
        self.z_mean = z_mean[0]
        self.l_mean = l_mean[0]
        self.w_mean = w_mean[0]
        self.t_mean = t_mean[0]
        self.x_std = x_std[0]
        self.y_std = y_std[0]
        self.z_std = z_std[0]
        self.l_std = l_std[0]
        self.w_std = w_std[0]
        self.t_std = t_std[0]

    def forward(self, x):
        for i in range(len(x)):
            for j in range(int(len(x[i]) / 6)):
                x[i][6 * j] = (((x[i][6 * j] - torch.Tensor(0.5)) * 340) - self.x_mean) / self.x_std
                x[i][6 * j + 1] = (((x[i][6 * j + 1] - 0.5) * 340) - self.y_mean) / self.y_std
                x[i][6 * j + 2] = (((x[i][6 * j + 2] - 0.5) * 600) - self.z_mean) / self.z_std
                x[i][6 * j + 3] = (((x[i][6 * j + 3] * 240) + 60) - self.l_mean) / self.l_std
                x[i][6 * j + 4] = (((x[i][6 * j + 4] * 144) + 6) - self.w_mean) / self.w_std
                x[i][6 * j + 5] = (((x[i][6 * j + 5] - 0.5) * 180) - self.t_mean) / self.t_std
                '''
                x[i][6 * j] = torch.clamp(x[i][6 * j], -2.021, 2.0113)
                x[i][6 * j+1] = torch.clamp(x[i][6 * j+1], -2.0488, 2.0655)
                x[i][6 * j + 2] = torch.clamp(x[i][6 * j + 2], -1.7734, 1.7538)
                x[i][6 * j + 3] = torch.clamp(x[i][6 * j + 3], -1.4848, 2.3212)
                x[i][6 * j + 4] = torch.clamp(x[i][6 * j + 4], -0.7424, 1.1247)
                x[i][6 * j + 4] = torch.clamp(x[i][6 * j + 4], -1.6584, 1.6643)
                '''
        return x


# Using tandem NN
class BackwardLSTM(nn.Module):
    def __init__(self, input_len, hidden_units, out_len, num_layers, activate_func):
        super(BackwardLSTM, self).__init__()

        # Ensure hidden_units and num_layers are lists
        assert isinstance(hidden_units, list), "hidden_units must be a list"
        assert isinstance(num_layers, list), "num_layers must be a list"
        assert len(hidden_units) == len(num_layers), "hidden_units and num_layers must have the same length"

        # Parameters
        self.input_len = input_len
        self.hidden_size = hidden_units
        self.out_len = out_len
        self.num_layers = num_layers
        self.hidden = None
        self.activate_func = activate_func
        #self.num_lstms = num_lstms

        # Layers
        '''
        self.encoder = nn.LSTM(input_size=input_len, hidden_size=hidden_units, num_layers=num_layers, batch_first=True)
        self.lstms = nn.ModuleList()
        for i in range(1, len(hidden_units)):
            self.lstms.append(
                nn.LSTM(input_size=hidden_units[i - 1], hidden_size=hidden_units[i], num_layers=num_layers[i],
                        batch_first=True))
        # self.lstms = get_clones(
        #    nn.LSTM(input_size=hidden_units, hidden_size=hidden_units, num_layers=num_layers, batch_first=True),
        #    num_lstms)
        '''
        self.encoder_decoder = EncoderDecoder(input_len=self.input_len, hidden_units=self.hidden_size,
                                              num_layers=self.num_layers, activate_func=self.activate_func,
                                              batch_first=True)
        self.feedforward = nn.Linear(in_features=hidden_units[-1], out_features=hidden_units[-1])
        self.fc1 = nn.Linear(in_features=hidden_units[-1], out_features=out_len)
        # self.clamp = Clamp(x_mean, y_mean, z_mean, l_mean, t_mean, x_std, y_std, z_std, l_std, t_std, device)

    def forward(self, x):
        '''
        modified_x = F.relu(self.encoder(x)[0])
        for i in range(len(self.hidden_size)):
            modified_x = F.relu(self.lstms[i](modified_x)[0])
        '''
        modified_x = self.encoder_decoder(x)
        # modified_x = F.relu(modified_x + self.feedforward(modified_x))  # residual
        out = self.fc1(modified_x)
        out = torch.tanh(out) * 2
        # out = self.clamp(out)
        return out, self.hidden


# Custom loss function
class RangeLoss(nn.Module):
    def __init__(self, ranges):
        super(RangeLoss, self).__init__()
        self.ranges = ranges

    def forward(self, output, target):
        mse_loss = nn.MSELoss()(output, target)
        penalty = 0.0
        for i, (min_val, max_val) in enumerate(self.ranges):
            penalty += torch.mean(torch.clamp(output[:, i] - max_val, min=0.0) ** 2)
            penalty += torch.mean(torch.clamp(min_val - output[:, i], min=0.0) ** 2)
        total_loss = mse_loss + penalty
        return total_loss


class CDLoss(nn.Module):
    def __init__(self, loss_fn, CD_loss_ratio=10):
        super(CDLoss, self).__init__()
        self.loss_fn = loss_fn
        self.CD_loss_ratio = CD_loss_ratio

    def forward(self, output, target):
        TL_TR_loss = self.loss_fn(output, target)
        CD_loss = self.CD_loss_ratio * self.loss_fn(torch.abs(output[:, 0:301] - output[:, 301:]),
                                                    torch.abs(target[:, 0:301] - target[:, 301:]))

        total_loss = TL_TR_loss + CD_loss
        return total_loss
