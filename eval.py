import torch
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
import os
import time

from data import create_dataset

from parameters import RESULTS_PATH, DATA_PATH, FIGS_PATH, MODEL_PATH, RODS, BATCH_SIZE, NUM_WORKERS, SAMPLE_RATE, \
    EPOCHS, DIM_MODEL, DIM_FEEDFORWARD, HEADS, NUM_LAYERS, START_TOKEN, END_TOKEN, PAD_TOKEN

device = "cuda" if torch.cuda.is_available() else "cpu"

