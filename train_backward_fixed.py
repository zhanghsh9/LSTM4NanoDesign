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
from models import BackwardLSTM
from train import train_epochs_backward
from parameters import RESULTS_PATH, DATA_PATH, FIGS_PATH, MODEL_PATH, RODS, BATCH_SIZE, NUM_WORKERS, SAMPLE_RATE, \
    LEARNING_RATE, EPOCHS, NUM_LAYERS, HIDDEN_UNITS, STEP_SIZE, GAMMA, ACTIVATE_FUNC

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
        device = torch.device('cuda:1')
        print(f'Running on {device} version = {torch.version.cuda}, device count = {torch.cuda.device_count()}')
        print()

    # mkdir
    timestamp = datetime.now().strftime('%Y%m%d')
    timestamp = '20240729_relu'
    RESULTS_PATH = os.path.join(RESULTS_PATH, 'backwards', 'fixed_attention')
    model_save_path = os.path.join(RESULTS_PATH, timestamp, MODEL_PATH)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    figs_save_path = os.path.join(RESULTS_PATH, timestamp, FIGS_PATH)
    if not os.path.exists(figs_save_path):
        os.mkdir(figs_save_path)

    shutil.copyfile('parameters.py', os.path.join(RESULTS_PATH, timestamp, 'parameters.py'))
    shutil.copyfile('train.py', os.path.join(RESULTS_PATH, timestamp, 'train.py'))
    shutil.copyfile('models.py', os.path.join(RESULTS_PATH, timestamp, 'models.py'))
    shutil.copyfile('train_forward_self_attention.py',
                    os.path.join(RESULTS_PATH, timestamp, 'train_forward_self_attention.py'))
    shutil.copyfile('data.py', os.path.join(RESULTS_PATH, timestamp, 'data.py'))
    if os.path.exists(os.path.join(RESULTS_PATH, timestamp, 'data')):
        shutil.rmtree(os.path.join(RESULTS_PATH, timestamp, 'data'))
    shutil.copytree('data', os.path.join(RESULTS_PATH, timestamp, 'data'))
    fixed_model_name = 'Forward_mse_vloss_best_attn_16.5.pth'
    shutil.copyfile(os.path.join('results/compare/fixed/models/', fixed_model_name),
                    os.path.join(model_save_path, 'fixed_attention_forward.pth'))

    # Set seed
    time_now = int(time.strftime("%Y%m%d%H%M%S", time.localtime()))
    torch.manual_seed(time_now)

    # Create dataset
    print('{}: Initializing dataset'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime())))
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset, test_dataset = create_dataset(data_path=DATA_PATH, rods=RODS, use_TL_TR=True, transform=transform,
                                                 sample_rate=SAMPLE_RATE, make_spectrum_int=False, device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)

    print('{}: Complete initializing dataset'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime())))
    print()

    # Create model
    input_len = train_dataset.max_src_seq_len
    out_len = train_dataset.max_tgt_seq_len
    # Forward
    print(f'{time.strftime("%Y%m%d  %H:%M:%S", time.localtime())}: Backward')
    x_mean = train_dataset.x_mean
    y_mean = train_dataset.y_mean
    z_mean = train_dataset.z_mean
    l_mean = train_dataset.l_mean
    t_mean = train_dataset.t_mean
    x_std = train_dataset.x_std
    y_std = train_dataset.y_std
    z_std = train_dataset.z_std
    l_std = train_dataset.l_std
    t_std = train_dataset.t_std
    backward_model_fixed = BackwardLSTM(input_len=out_len, hidden_units=HIDDEN_UNITS, out_len=input_len,
                                        num_layers=NUM_LAYERS, activate_func=ACTIVATE_FUNC, x_mean=x_mean,
                                        y_mean=y_mean, z_mean=z_mean, l_mean=l_mean, t_mean=t_mean,
                                        x_std=x_std, y_std=y_std, z_std=z_std, l_std=l_std, t_std=t_std,
                                        device=device).to(device)

    for p in backward_model_fixed.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # This code is very important! It initialises the parameters with a range of values that stops the signal
    # fading or getting too big.
    # See https://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization for a mathematical
    # explanation.

    backward_loss_fn_MSE = MSELoss(reduction='mean').to(device)
    backward_optimizer_Adam_fixed = Adam(params=backward_model_fixed.parameters(), lr=LEARNING_RATE)

    # See https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
    backward_step_lr_fixed = StepLR(optimizer=backward_optimizer_Adam_fixed, step_size=STEP_SIZE, gamma=GAMMA)

    # Load forward models
    forward_model_fixed = torch.load(os.path.join(model_save_path, 'fixed_attention_forward.pth'))
    forward_model_fixed = forward_model_fixed.to(device)

    # Train
    backward_model_fixed, x_axis_loss_fixed, x_axis_vloss_fixed, loss_record_fixed, vloss_record_fixed = train_epochs_backward(
        training_loader=train_dataloader, test_loader=test_dataloader, forward_model=forward_model_fixed,
        backward_model=backward_model_fixed, loss_fn=backward_loss_fn_MSE, optimizer=backward_optimizer_Adam_fixed,
        scheduler=backward_step_lr_fixed, timestamp=timestamp, epochs=EPOCHS, results_path=RESULTS_PATH, device=device)

    # Save model
    model_name = f'Backward_epochs_{EPOCHS}_lstms_{len(HIDDEN_UNITS)}_hidden_{HIDDEN_UNITS}_fixed.pth'
    if os.path.exists(os.path.join(model_save_path, model_name)):
        os.remove(os.path.join(model_save_path, model_name))
    torch.save(backward_model_fixed, os.path.join(model_save_path, model_name))

    plt.figure()
    figs_name = 'loss_backward_fixed.png'
    # plt.axes(yscale="log")
    plt1, = plt.plot(x_axis_loss_fixed, loss_record_fixed, label='loss')
    plt2, = plt.plot(x_axis_vloss_fixed, vloss_record_fixed, label='vloss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.title('Loss to epochs, backward')
    if os.path.exists(os.path.join(figs_save_path, figs_name)):
        os.remove(os.path.join(figs_save_path, figs_name))
    plt.savefig(os.path.join(figs_save_path, figs_name))
    plt.show()
    plt.close()

    loss_save = {'loss_record_fixed': loss_record_fixed, 'vloss_record_fixed': vloss_record_fixed,
                 'seed': time_now, 'EPOCHS': EPOCHS, 'BATCH_SIZE': BATCH_SIZE,
                 'NUM_LAYERS': NUM_LAYERS, 'LEARNING_RATE': LEARNING_RATE, 'STEP_SIZE': STEP_SIZE,
                 'GAMMA': GAMMA}
    scio.savemat(os.path.join(RESULTS_PATH, timestamp, 'loss.mat'), mdict=loss_save)

    end_time = time.time()
    print()
    print('{}: Total time used: {}'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime()),
                                           time.strftime('%d days %H h %M m %S s ', time.gmtime(end_time - start_time))))
