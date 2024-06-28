import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import MSELoss
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
from models import ForwardPredictionLSTM, BackwardPredictionLSTM
# from train import train_epochs_forward, train_epochs_backward
from parameters import RESULTS_PATH, DATA_PATH, FIGS_PATH, MODEL_PATH, RODS, BATCH_SIZE, NUM_WORKERS, SAMPLE_RATE, \
    LEARNING_RATE, EPOCHS, NUM_LAYERS, ATTENTION, HIDDEN_UNITS, STEP_SIZE, GAMMA

if __name__ == '__main__':
    print(f'{time.strftime("%Y%m%d  %H:%M:%S", time.localtime())}: Begin')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    time_now = int(time.strftime("%Y%m%d%H%M%S", time.localtime()))
    torch.manual_seed(time_now)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset, test_dataset = create_dataset(data_path=DATA_PATH, rods=RODS, use_TL_TR=True, transform=transform,
                                                 sample_rate=SAMPLE_RATE, make_spectrum_int=False, device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)
    print('{}: Using dataset:'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime())))
    # print()
    print('Train:')
    train_dataset.print()
    train_dataset.print_item(0)
    print()
    print('Test:')
    test_dataset.print()
    print('{}: Complete initializing dataset'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime())))
    print()
    print(f'{time.strftime("%Y%m%d  %H:%M:%S", time.localtime())}: Finished')

