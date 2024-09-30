import torch
import torch.nn.functional as F
from torch.nn import MSELoss, L1Loss
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import scipy.io as scio

from data import create_dataset

from parameters import RESULTS_PATH, DATA_PATH, FIGS_PATH, MODEL_PATH, RODS, NUM_WORKERS, SAMPLE_RATE

if __name__ == '__main__':
    # Get device
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    else:
        device = torch.device('cuda:2')
        print(f'Running on {device} version = {torch.version.cuda}, device count = {torch.cuda.device_count()}')
        print()

    time_now = int(time.strftime("%Y%m%d%H%M%S", time.localtime()))
    torch.manual_seed(time_now)

    # dir
    timestamp = '20240914_leakyrelu'
    RESULTS_PATH = os.path.join(RESULTS_PATH, 'backwards/self_attention')
    model_save_path = os.path.join(RESULTS_PATH, timestamp, MODEL_PATH)
    figs_save_path = os.path.join(RESULTS_PATH, timestamp, FIGS_PATH)

    # Create dataloader
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset, test_dataset = create_dataset(data_path=DATA_PATH, rods=RODS, use_TL_TR='TL_TR', transform=transform,
                                     sample_rate=SAMPLE_RATE, make_spectrum_int=False, device=device)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, drop_last=True,
                                 pin_memory=True)
    # Forward prediction
    forward_loss_fn = MSELoss()
    forward_loss_fn_MAE = L1Loss(reduction='mean')

    lamda = range(900, 1801, 3 * SAMPLE_RATE)

    # Load models
    # model_name = f'Forward_epochs_{EPOCHS}_lstms_{len(HIDDEN_UNITS)}_hidden_{HIDDEN_UNITS}.pth'
    forward_model_name = f'self_attention_forward.pth'
    forward_model = torch.load(os.path.join(model_save_path, forward_model_name))
    forward_model.to(device)
    forward_model.eval()
    backward_model_name = f'Backward_mse_vloss_best.pth'
    backward_model = torch.load(os.path.join(model_save_path, backward_model_name))
    backward_model.to(device)
    backward_model.eval()
    backward_mse_loss_sum = 0
    backward_mae_loss_sum = 0
    prediction = []
    target = []
    tandem_output = []
    vloss_best = 100

    # Get normalize parameters
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

    data=scio.loadmat(os.path.join(DATA_PATH, 'gaussian_spectrum.mat'))
    gaussian_spectrum=torch.Tensor(data['T']).to(device)

    with torch.no_grad():
        for i in range(len(gaussian_spectrum)):
            voutput_backward, _ = backward_model(torch.unsqueeze(torch.cat((gaussian_spectrum[i], gaussian_spectrum[i]), dim=0),dim=0))

            voutput_backward = voutput_backward.to('cpu').view(-1).tolist()
            inverse_normalize = []
            for j in range(int(len(voutput_backward) / 6)):
                inverse_normalize.append(voutput_backward[6 * j] * x_std + x_mean)
                inverse_normalize.append(voutput_backward[6 * j + 1] * y_std + y_mean)
                inverse_normalize.append(voutput_backward[6 * j + 2] * z_std + z_mean)
                inverse_normalize.append(voutput_backward[6 * j + 3] * l_std + l_mean)
                r = abs(voutput_backward[6 * j + 3] / voutput_backward[6 * j + 4])  # r = l / w
                inverse_normalize.append(inverse_normalize[-1] / r)  # w = l / r
                inverse_normalize.append(voutput_backward[6 * j + 5] * t_std + t_mean)
            prediction.append(inverse_normalize)

    # print(forward_model.self_attention.weight)
    results_save = {'prediction': prediction}
    scio.savemat(os.path.join(RESULTS_PATH, timestamp, 'results_gaussian.mat'), mdict=results_save)
