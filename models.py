import torch
from torch import nn
import torch.nn.functional as F

import math
import copy
from datetime import datetime
import time


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


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
    def __init__(self, attention, input_len, hidden_units, out_len, num_layers):
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

        # Layers
        self.fixed_attention = FixedAttention(attention)
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
        self.feedforward = nn.Linear(in_features=hidden_units[-1], out_features=hidden_units[-1], bias=True)
        self.fc1 = nn.Linear(in_features=hidden_units[-1], out_features=out_len, bias=True)

    def forward(self, x):
        attentioned_x = self.fixed_attention(x)
        modified_x = F.relu(self.encoder(attentioned_x)[0])
        for i in range(len(self.hidden_size) - 1):
            modified_x = F.relu(self.lstms[i](modified_x)[0])

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
    def __init__(self, input_len, hidden_units, out_len, num_layers):
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

        # Layers
        self.self_attention = SelfAttention(input_len=self.input_len, out_len=self.input_len)
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
        attentioned_x = self.self_attention(x)
        modified_x = F.relu(self.encoder(attentioned_x)[0])
        for i in range(len(self.hidden_size) - 1):
            modified_x = F.relu(self.lstms[i](modified_x)[0])

        # modified_x = F.relu(modified_x + self.feedforward(modified_x))  # residual
        out = self.fc1(modified_x)
        out = torch.sigmoid(out)
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
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.feature_dim ** 0.5)
        attn_weights = self.softmax(attn_scores)

        # Compute weighted sum
        output = torch.matmul(attn_weights, V)
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
    def __init__(self, input_len, hidden_units, out_len, num_layers, num_heads):
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
        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.hidden_size,
                                                         num_heads=self.num_heads, batch_first=True)
        self.feedforward = nn.Linear(in_features=hidden_units[-1], out_features=hidden_units[-1], bias=True)
        self.fc1 = nn.Linear(in_features=hidden_units[-1], out_features=out_len, bias=True)

    def forward(self, x):
        modified_x = F.relu(self.encoder(x)[0])
        for i in range(len(self.hidden_size) - 1):
            modified_x = F.relu(self.lstms[i](modified_x)[0])
        # modified_x = F.relu(modified_x + self.feedforward(modified_x))  # residual
        attentioned_x, _ = self.multihead_attention(modified_x, modified_x, modified_x)
        # attentioned_x shape: (batch_size, seq_length, hidden_dim)
        # Selecting the output corresponding to the last sequence element
        out = self.fc1(attentioned_x[:, -1, :])
        # out = self.fc1(modified_x)
        out = torch.sigmoid(out)
        return out, self.hidden


# Using tandem NN
class BackwardPredictionLSTM(nn.Module):
    def __init__(self, input_len, hidden_units, out_len, num_layers):
        super(BackwardPredictionLSTM, self).__init__()

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
        #self.num_lstms = num_lstms

        # Layers
        self.encoder = nn.LSTM(input_size=input_len, hidden_size=hidden_units, num_layers=num_layers, batch_first=True)
        self.lstms = nn.ModuleList()
        for i in range(1, len(hidden_units)):
            self.lstms.append(
                nn.LSTM(input_size=hidden_units[i - 1], hidden_size=hidden_units[i], num_layers=num_layers[i],
                        batch_first=True))
        # self.lstms = get_clones(
        #    nn.LSTM(input_size=hidden_units, hidden_size=hidden_units, num_layers=num_layers, batch_first=True),
        #    num_lstms)
        self.feedforward = nn.Linear(in_features=hidden_units[-1], out_features=hidden_units[-1])
        self.fc1 = nn.Linear(in_features=hidden_units[-1], out_features=out_len)

    def forward(self, x):
        modified_x = F.relu(self.encoder(x)[0])
        for i in range(len(self.hidden_size)):
            modified_x = F.relu(self.lstms[i](modified_x)[0])

        # modified_x = F.relu(modified_x + self.feedforward(modified_x))  # residual
        out = self.fc1(modified_x)
        return out, self.hidden
