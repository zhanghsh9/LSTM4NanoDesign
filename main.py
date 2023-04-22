import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss, NLLLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np
import os
from datetime import datetime
import time
import shutil

from data import create_dataset
from models import Transformer, Transformer2
from train import train_epochs_transformer
from parameters import RESULTS_PATH, DATA_PATH, FIGS_PATH, MODEL_PATH, RODS, BATCH_SIZE, NUM_WORKERS, SAMPLE_RATE, \
    LEARNING_RATE, EPOCHS, NUM_LAYERS, HEADS, DIM_MODEL, DIM_FEEDFORWARD, DROPOUT, NUM_WORDS, MAX_SEQ_LEN, PAD_TOKEN, \
    TF_RATIO


