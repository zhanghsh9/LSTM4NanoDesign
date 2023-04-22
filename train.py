import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import MSELoss

from datetime import datetime
import time
import os
import random

from eval import greedy_decoder
from parameters import EPOCHS, VALID_FREQ, PAD_TOKEN, HEADS, RESULTS_PATH, MODEL_PATH, START_TOKEN, END_TOKEN, \
    NUM_WORDS, TF_RATIO

