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
from models import ForwardPredictionLSTM
from train import train_epochs_forward
from parameters import RESULTS_PATH, DATA_PATH, FIGS_PATH, MODEL_PATH, RODS, BATCH_SIZE, NUM_WORKERS, SAMPLE_RATE, \
    LEARNING_RATE, EPOCHS, NUM_LAYERS, DROPOUT, ATTENTION, HIDDEN_UNITS, NUM_LSTMS, STEP_SIZE, GAMMA

if __name__ == '__main__':
    start_time = time.time()
    # Loss record
    forward_loss_rec = []
    forward_vloss_rec = []
    backward_loss_rec = []
    backward_vloss_rec = []

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print('Running on {}'.format(device))
    print()

    # mkdir
    timestamp = datetime.now().strftime('%Y%m%d')
    model_save_path = os.path.join(RESULTS_PATH, timestamp, MODEL_PATH)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    figs_save_path = os.path.join(RESULTS_PATH, timestamp, FIGS_PATH)
    if not os.path.exists(figs_save_path):
        os.mkdir(figs_save_path)

    shutil.copyfile('parameters.py', os.path.join(RESULTS_PATH, timestamp, 'parameters.py'))

    # Set seed
    time_now = int(time.strftime("%Y%m%d%H%M%S", time.localtime()))
    torch.manual_seed(time_now)

    # Create dataset
    print('{}: Initializing dataset'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime())))
    transform = transforms.Compose([transforms.ToTensor()])
    forward_train_dataset, forward_test_dataset = create_dataset(data_path=DATA_PATH, rods=RODS, reverse=False,
                                                                 use_TL=True, transform=transform,
                                                                 sample_rate=SAMPLE_RATE)
    forward_train_dataloader = DataLoader(forward_train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                          num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)
    forward_test_dataloader = DataLoader(forward_test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                         num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)

    print('{}: Using dataset:'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime())))
    print()
    print('Train:')
    forward_train_dataset.print()
    forward_train_dataset.print_item(0)
    print()
    print('Test:')
    forward_test_dataset.print()
    print('{}: Complete initializing dataset'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime())))
    print()

    for ATTENTION in range(1, 11, 2):

        print('ATTENTION = {}'.format(ATTENTION))

        # Create model
        input_len = forward_train_dataset.max_src_seq_len
        out_len = forward_train_dataset.max_tgt_seq_len

        model = ForwardPredictionLSTM(attention=ATTENTION, input_len=input_len, hidden_units=HIDDEN_UNITS,
                                      out_len=out_len, num_layers=NUM_LAYERS, num_lstms=NUM_LSTMS).to(device)

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # This code is very important! It initialises the parameters with a range of values that stops the signal fading or
        # getting too big.
        # See https://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
        # for a mathematical explanation.

        loss_fn_MSE = MSELoss().to(device)
        forward_optimizer_Adam = Adam(model.parameters(), lr=LEARNING_RATE)

        # See https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
        step_lr = StepLR(forward_optimizer_Adam, step_size=STEP_SIZE, gamma=GAMMA, verbose=True)

        # Train
        model, x_axis_loss, x_axis_vloss, loss_record, vloss_record = train_epochs_forward(
            training_loader=forward_train_dataloader, test_loader=forward_test_dataloader, model=model,
            loss_fn=loss_fn_MSE, optimizer=forward_optimizer_Adam,
            scheduler=step_lr, epochs=EPOCHS)

        # Save model
        model_name = 'Forward_epochs_{}_lstms_{}_hidden_{}_attn_{}.pth'.format(EPOCHS, NUM_LSTMS, HIDDEN_UNITS,
                                                                               ATTENTION)
        if os.path.exists(os.path.join(model_save_path, model_name)):
            os.remove(os.path.join(model_save_path, model_name))
        torch.save(model, os.path.join(model_save_path, model_name))

        # Draw loss figure
        figs_name = 'loss_forward_attn_{}.png'.format(ATTENTION)
        # plt.axes(yscale="log")
        plt1, = plt.plot(x_axis_loss, loss_record, label='loss')
        plt2, = plt.plot(x_axis_vloss, vloss_record, label='vloss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.title('Loss to epochs, forward')
        if os.path.exists(os.path.join(figs_save_path, figs_name)):
            os.remove(os.path.join(figs_save_path, figs_name))
        plt.savefig(os.path.join(figs_save_path, figs_name))
        plt.show()

        loss_save = {'loss_record': loss_record, 'vloss_record': vloss_record, 'seed': time_now, 'EPOCHS': EPOCHS,
                     'BATCH_SIZE': BATCH_SIZE, 'NUM_LAYERS': NUM_LAYERS, 'NUM_LSTMS': NUM_LSTMS,
                     'LEARNING_RATE': LEARNING_RATE, 'STEP_SIZE': STEP_SIZE, 'GAMMA': GAMMA}
        scio.savemat(os.path.join(RESULTS_PATH, timestamp, 'loss_attn{}.mat'.format(ATTENTION)), mdict=loss_save)

        end_time = time.time()
        print()
        print('{}: Total time used: {}'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime()),
                                               time.strftime('%H h %M m %S s ', time.gmtime(end_time - start_time))))
