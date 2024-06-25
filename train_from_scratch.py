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
from train import train_epochs_forward, train_epochs_backward
from parameters import RESULTS_PATH, DATA_PATH, FIGS_PATH, MODEL_PATH, RODS, BATCH_SIZE, NUM_WORKERS, SAMPLE_RATE, \
    LEARNING_RATE, EPOCHS, NUM_LAYERS, ATTENTION, HIDDEN_UNITS, STEP_SIZE, GAMMA

if __name__ == '__main__':
    start_time = time.time()
    # Loss record
    forward_loss_rec = []
    forward_vloss_rec = []
    backward_loss_rec = []
    backward_vloss_rec = []

    # Get device
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    else:
        device = torch.device('cuda')
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = "cpu"
        print(f'Running on {device} version = {torch.version.cuda}')
        print()

    # mkdir
    timestamp = datetime.now().strftime('%Y%m%d')
    # timestamp = '20230426'
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
    train_dataset, test_dataset = create_dataset(data_path=DATA_PATH, rods=RODS, use_TL=True, transform=transform,
                                                 sample_rate=SAMPLE_RATE)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)

    print('{}: Using dataset:'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime())))
    print()
    print('Train:')
    train_dataset.print()
    train_dataset.print_item(0)
    print()
    print('Test:')
    test_dataset.print()
    print('{}: Complete initializing dataset'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime())))
    print()

    for ATTENTION in [4.5, 5.5]:

        print('ATTENTION = {}'.format(ATTENTION))

        # Create model
        input_len = train_dataset.max_src_seq_len
        out_len = train_dataset.max_tgt_seq_len

        # Forward
        print('{}: Forward'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime())))
        forward_model = ForwardPredictionLSTM(attention=ATTENTION, input_len=input_len, hidden_units=HIDDEN_UNITS,
                                              out_len=out_len, num_layers=NUM_LAYERS, num_lstms=NUM_LSTMS).to(device)

        for p in forward_model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # This code is very important! It initialises the parameters with a range of values that stops the signal
        # fading or getting too big.
        # See https://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization for a mathematical
        # explanation.

        forward_loss_fn_MSE = MSELoss().to(device)
        forward_optimizer_Adam = Adam(forward_model.parameters(), lr=LEARNING_RATE)

        # See https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
        forward_step_lr = StepLR(forward_optimizer_Adam, step_size=STEP_SIZE, gamma=GAMMA, verbose=True)

        # Train
        forward_model, x_axis_loss, x_axis_vloss, loss_record, vloss_record = train_epochs_forward(
            training_loader=train_dataloader, test_loader=test_dataloader, model=forward_model,
            loss_fn=forward_loss_fn_MSE, optimizer=forward_optimizer_Adam, scheduler=forward_step_lr,
            attention=ATTENTION, timestamp=timestamp, epochs=EPOCHS)

        # Save model
        model_name = 'Forward_epochs_{}_lstms_{}_hidden_{}_attn_{}.pth'.format(EPOCHS, NUM_LSTMS, HIDDEN_UNITS,
                                                                               ATTENTION)
        if os.path.exists(os.path.join(model_save_path, model_name)):
            os.remove(os.path.join(model_save_path, model_name))
        torch.save(forward_model, os.path.join(model_save_path, model_name))

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

        # Backward
        start_time = time.time()
        print('{}: Backward'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime())))
        backward_model = BackwardPredictionLSTM(input_len=out_len, hidden_units=HIDDEN_UNITS, out_len=input_len,
                                                num_layers=NUM_LAYERS, num_lstms=NUM_LSTMS).to(device)

        for p in backward_model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        backward_loss_fn_MSE = MSELoss().to(device)
        backward_optimizer_Adam = Adam(backward_model.parameters(), lr=LEARNING_RATE)

        backward_step_lr = StepLR(backward_optimizer_Adam, step_size=STEP_SIZE, gamma=GAMMA, verbose=True)

        backward_model, x_axis_loss, x_axis_vloss, loss_record, vloss_record = train_epochs_backward(
            training_loader=train_dataloader, test_loader=test_dataloader, backward_model=backward_model,
            loss_fn=backward_loss_fn_MSE, optimizer=backward_optimizer_Adam, scheduler=backward_step_lr,
            attention=ATTENTION, timestamp=timestamp, epochs=EPOCHS)

        model_name = 'Backward_epochs_{}_lstms_{}_hidden_{}_attn_{}.pth'.format(EPOCHS, NUM_LSTMS, HIDDEN_UNITS,
                                                                                ATTENTION)
        if os.path.exists(os.path.join(model_save_path, model_name)):
            os.remove(os.path.join(model_save_path, model_name))
        torch.save(backward_model, os.path.join(model_save_path, model_name))

        # Draw loss figure
        figs_name = 'loss_backward_attn_{}.png'.format(ATTENTION)
        # plt.axes(yscale="log")
        plt1, = plt.plot(x_axis_loss, loss_record, label='loss')
        plt2, = plt.plot(x_axis_vloss, vloss_record, label='vloss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.title('Loss to epochs, backward')
        if os.path.exists(os.path.join(figs_save_path, figs_name)):
            os.remove(os.path.join(figs_save_path, figs_name))
        plt.savefig(os.path.join(figs_save_path, figs_name))
        plt.show()

        loss_save = {'loss_record': loss_record, 'vloss_record': vloss_record, 'seed': time_now, 'EPOCHS': EPOCHS,
                     'BATCH_SIZE': BATCH_SIZE, 'NUM_LAYERS': NUM_LAYERS, 'NUM_LSTMS': NUM_LSTMS,
                     'LEARNING_RATE': LEARNING_RATE, 'STEP_SIZE': STEP_SIZE, 'GAMMA': GAMMA}
        scio.savemat(os.path.join(RESULTS_PATH, timestamp, 'loss_attn_backward{}.mat'.format(ATTENTION)),
                     mdict=loss_save)

        end_time = time.time()
        print()
        print('{}: Total time used: {}'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime()),
                                               time.strftime('%H h %M m %S s ', time.gmtime(end_time - start_time))))
