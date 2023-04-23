import torch
from torch import nn
import torch.nn.functional as F

import math
import copy
from datetime import datetime
import time

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    def __init__(self, attention, input_len, hidden_units, out_len, num_layers, num_lstms):
        super(ForwardPredictionLSTM, self).__init__()

        # Parameters
        self.hidden_size = hidden_units
        self.num_layers = num_layers
        self.hidden = None
        self.num_lstms = num_lstms

        # Layers
        self.fixed_attention = FixedAttention(attention)
        self.encoder = nn.LSTM(input_size=input_len, hidden_size=hidden_units, num_layers=num_layers, batch_first=True)
        self.lstms = get_clones(
            nn.LSTM(input_size=hidden_units, hidden_size=hidden_units, num_layers=num_layers, batch_first=True),
            num_lstms)
        self.feedforward = nn.Linear(hidden_units, hidden_units)
        self.fc1 = nn.Linear(hidden_units, out_len)

    def forward(self, x):
        modified_x = self.fixed_attention(x)
        modified_x = F.relu(self.encoder(modified_x)[0])
        for i in range(self.num_lstms):
            modified_x = F.relu(self.lstms[i](modified_x)[0])

        modified_x = F.relu(modified_x + self.feedforward(modified_x))  # residual
        out = self.fc1(modified_x)
        return out, self.hidden


# Using tandem NN
class BackwardPredictionLSTM(nn.Module):
    def __init__(self, attention, input_len, hidden_units, out_len, num_layers, num_lstms):
        super(BackwardPredictionLSTM, self).__init__()

        # Parameters
        self.hidden_size = hidden_units
        self.num_layers = num_layers
        self.hidden = None
        self.num_lstms = num_lstms

        # Layers
        self.encoder = nn.LSTM(input_size=input_len, hidden_size=hidden_units, num_layers=num_layers, batch_first=True)
        self.lstms = get_clones(
            nn.LSTM(input_size=hidden_units, hidden_size=hidden_units, num_layers=num_layers, batch_first=True),
            num_lstms)
        self.feedforward = nn.Linear(hidden_units, hidden_units)
        self.fc1 = nn.Linear(hidden_units, out_len)

    def forward(self, x):
        modified_x = F.relu(self.encoder(x)[0])
        for i in range(self.num_lstms):
            modified_x = F.relu(self.lstms[i](modified_x)[0])

        modified_x = F.relu(modified_x + self.feedforward(modified_x))  # residual
        out = self.fc1(modified_x)
        return out, self.hidden
