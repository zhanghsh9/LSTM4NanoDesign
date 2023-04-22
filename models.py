import torch
from torch import nn
import torch.nn.functional as F
import math
import copy

from parameters import DIM_MODEL, NUM_LAYERS, HEADS, DROPOUT, DIM_FEEDFORWARD, NUM_WORDS, MAX_SEQ_LEN, PAD_TOKEN

device = "cuda" if torch.cuda.is_available() else "cpu"