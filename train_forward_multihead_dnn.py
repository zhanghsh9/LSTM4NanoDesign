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
from models import ForwardMultiheadAttentionDNN
from train import train_epochs_forward
from parameters import RESULTS_PATH, DATA_PATH, FIGS_PATH, MODEL_PATH, RODS, BATCH_SIZE, NUM_WORKERS, SAMPLE_RATE, \
    LEARNING_RATE, EPOCHS, NUM_LAYERS, HIDDEN_UNITS, STEP_SIZE, GAMMA, ACTIVATE_FUNC, NUM_HEADS

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
        device = torch.device('cuda:3')
        print(f'Running on {device} version = {torch.version.cuda}, device count = {torch.cuda.device_count()}')
        print()

    # mkdir
    timestamp = datetime.now().strftime('%Y%m%d')
    timestamp = '20240929_relu'
    RESULTS_PATH = os.path.join(RESULTS_PATH, 'multihead_attention_dnn')
    model_save_path = os.path.join(RESULTS_PATH, timestamp, MODEL_PATH)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    figs_save_path = os.path.join(RESULTS_PATH, timestamp, FIGS_PATH)
    if not os.path.exists(figs_save_path):
        os.mkdir(figs_save_path)

    shutil.copyfile('parameters.py', os.path.join(RESULTS_PATH, timestamp, 'parameters.py'))
    shutil.copyfile('train.py', os.path.join(RESULTS_PATH, timestamp, 'train.py'))
    shutil.copyfile('models.py', os.path.join(RESULTS_PATH, timestamp, 'models.py'))
    shutil.copyfile('train_forward_multihead_dnn.py',
                    os.path.join(RESULTS_PATH, timestamp, 'train_forward_multihead_dnn.py'))
    shutil.copyfile('data.py', os.path.join(RESULTS_PATH, timestamp, 'data.py'))
    if os.path.exists(os.path.join(RESULTS_PATH, timestamp, 'data')):
        shutil.rmtree(os.path.join(RESULTS_PATH, timestamp, 'data'))
    shutil.copytree('data', os.path.join(RESULTS_PATH, timestamp, 'data'))

    # Set seed
    time_now = int(time.strftime("%Y%m%d%H%M%S", time.localtime()))
    torch.manual_seed(time_now)

    # Create dataset
    print('{}: Initializing dataset'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime())))
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset, test_dataset = create_dataset(data_path=DATA_PATH, rods=RODS, use_TL_TR='TL_TR',
                                                 transform=transform, sample_rate=SAMPLE_RATE, make_spectrum_int=False,
                                                 device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)
    '''
    print('{}: Using dataset:'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime())))
    print()
    print('Train:')
    train_dataset.print()
    train_dataset.print_item(0)
    print()
    print('Test:')
    test_dataset.print()
    test_dataset.print_item(0)
    '''

    print('{}: Complete initializing dataset'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime())))
    print()

    # Create model
    input_len = train_dataset.max_src_seq_len
    out_len = train_dataset.max_tgt_seq_len
    # Forward
    print(f'{time.strftime("%Y%m%d  %H:%M:%S", time.localtime())}: Forward')
    forward_model = ForwardMultiheadAttentionDNN(input_len=input_len, hidden_units=HIDDEN_UNITS, out_len=out_len,
                                                 activate_func=ACTIVATE_FUNC, num_heads=NUM_HEADS).to(device)

    for p in forward_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # This code is very important! It initialises the parameters with a range of values that stops the signal
    # fading or getting too big.
    # See https://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization for a mathematical
    # explanation.

    forward_loss_fn = MSELoss(reduction='mean').to(device)
    forward_optimizer_Adam = Adam(params=forward_model.parameters(), lr=LEARNING_RATE)

    # See https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
    forward_step_lr = StepLR(optimizer=forward_optimizer_Adam, step_size=STEP_SIZE, gamma=GAMMA)

    # Train
    forward_model, x_axis_loss, x_axis_vloss, loss_record, vloss_record = train_epochs_forward(
        training_loader=train_dataloader, test_loader=test_dataloader, model=forward_model,
        loss_fn=forward_loss_fn, optimizer=forward_optimizer_Adam, scheduler=forward_step_lr,
        attention=0, timestamp=timestamp, epochs=EPOCHS, results_path=RESULTS_PATH, device=device)

    # Save model
    model_name = f'Forward_epochs_{EPOCHS}_dnn_{len(HIDDEN_UNITS)}_hidden_{HIDDEN_UNITS}.pth'
    if os.path.exists(os.path.join(model_save_path, model_name)):
        os.remove(os.path.join(model_save_path, model_name))
    torch.save(forward_model, os.path.join(model_save_path, model_name))

    # Draw loss figure
    plt.figure()
    figs_name = 'loss_forward.png'
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
    plt.close()

    loss_save = {'loss_record': loss_record, 'vloss_record': vloss_record, 'seed': time_now, 'EPOCHS': EPOCHS,
                 'BATCH_SIZE': BATCH_SIZE, 'NUM_LAYERS': NUM_LAYERS, 'LEARNING_RATE': LEARNING_RATE,
                 'STEP_SIZE': STEP_SIZE, 'GAMMA': GAMMA, 'x_axis_loss': x_axis_loss, 'x_axis_vloss': x_axis_vloss,
                 'time_used': start_time - time.time()}
    scio.savemat(os.path.join(RESULTS_PATH, timestamp, 'loss.mat'), mdict=loss_save)

    end_time = time.time()
    print()
    print('{}: Total time used: {}'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime()),
                                           time.strftime('%H h %M m %S s ', time.gmtime(end_time - start_time))))
