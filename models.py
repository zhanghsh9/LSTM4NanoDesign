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


class ForwardPredictionLSTM(nn.Module):
    def __init__(self, attention, input_len, hidden_units, out_len, num_layers):
        super(ForwardPredictionLSTM, self).__init__()

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
        for i in range(len(self.num_lstms)):
            modified_x = F.relu(self.lstms[i](modified_x)[0])

        # modified_x = F.relu(modified_x + self.feedforward(modified_x))  # residual
        out = self.fc1(modified_x)
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
        for i in range(len(self.num_lstms)):
            modified_x = F.relu(self.lstms[i](modified_x)[0])

        # modified_x = F.relu(modified_x + self.feedforward(modified_x))  # residual
        out = self.fc1(modified_x)
        return out, self.hidden
