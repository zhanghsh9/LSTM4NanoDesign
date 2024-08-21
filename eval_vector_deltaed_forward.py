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

from data import create_dataset_delta

from parameters import RESULTS_PATH, DATA_PATH, FIGS_PATH, MODEL_PATH, RODS, BATCH_SIZE, NUM_WORKERS, SAMPLE_RATE, \
    EPOCHS, NUM_LAYERS, HIDDEN_UNITS

if __name__ == '__main__':
    # Get device
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    else:
        device = torch.device('cuda:1')
        print(f'Running on {device} version = {torch.version.cuda}, device count = {torch.cuda.device_count()}')
        print()

    time_now = int(time.strftime("%Y%m%d%H%M%S", time.localtime()))
    torch.manual_seed(time_now)

    # dir
    timestamp = '20240820_relu'
    RESULTS_PATH = os.path.join(RESULTS_PATH, 'vector_attention/deltaed')
    model_save_path = os.path.join(RESULTS_PATH, timestamp, MODEL_PATH)
    figs_save_path = os.path.join(RESULTS_PATH, timestamp, FIGS_PATH)

    # Create dataloader
    transform = transforms.Compose([transforms.ToTensor()])
    _, test_dataset = create_dataset_delta(data_path=os.path.join(RESULTS_PATH, timestamp, 'data'), rods=RODS,
                                           use_TL_TR=True, transform=transform, sample_rate=SAMPLE_RATE,
                                           make_spectrum_int=False, device=device)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, drop_last=True,
                                 pin_memory=True)
    # Forward prediction
    forward_model = []
    forward_loss_fn = MSELoss()
    forward_loss_fn_MAE = L1Loss()

    lamda = range(900, 1801, 3 * SAMPLE_RATE)

    # Load models
    # model_name = f'Forward_epochs_{EPOCHS}_lstms_{len(HIDDEN_UNITS)}_hidden_{HIDDEN_UNITS}.pth'
    model_name = f'Forward_mse_vloss_best_attn_0.pth'
    forward_model = torch.load(os.path.join(model_save_path, model_name))
    forward_model.to(device)
    forward_model.eval()
    forward_mse_loss_sum = 0
    forward_mae_loss_sum = 0
    prediction = []
    real = []
    vloss_best = 100

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            vinputs, vlabels = data
            vinputs, vlabels = vinputs.float().to(device), vlabels.float().to(device)
            voutput, _ = forward_model(vinputs)
            mse_loss = forward_loss_fn(vlabels, voutput).item()
            mae_loss = forward_loss_fn_MAE(vlabels, voutput).item()
            forward_mse_loss_sum = forward_mse_loss_sum + mse_loss
            forward_mae_loss_sum = forward_mae_loss_sum + mae_loss
            prediction.append(voutput.to('cpu').tolist())
            real.append(vlabels.to('cpu').tolist())
            if mse_loss < vloss_best:
                plt.figure()
                plt1, = plt.plot(lamda, vlabels[0, 0:301].cpu(), label='Real')
                plt2, = plt.plot(lamda, voutput[0, 0:301].cpu(), label=f'Predict, MSE = {mse_loss * 100:.2f}%')
                plt.legend()
                plt.xlabel('lambda(nm)')
                plt.ylabel('TL')
                plt.title('Forward')

                if os.path.exists(os.path.join(figs_save_path, f'TL_forward_best.png')):
                    os.remove(os.path.join(figs_save_path, f'TL_forward_attn_best.png'))
                plt.savefig(os.path.join(figs_save_path, f'TL_forward_attn_best.png'))
                plt.close()

                # Forward, TR
                plt.figure()
                plt1, = plt.plot(lamda, vlabels[0, 301:].cpu(), label='Real')
                plt2, = plt.plot(lamda, voutput[0, 301:].cpu(), label=f'Predict, MSE = {mse_loss * 100:.2f}%')
                plt.legend()
                plt.xlabel('lambda(nm)')
                plt.ylabel('TR')
                plt.title('Forward')

                if os.path.exists(os.path.join(figs_save_path, f'TR_forward_attn_best.png')):
                    os.remove(os.path.join(figs_save_path, f'TR_forward_attn_best.png'))
                plt.savefig(os.path.join(figs_save_path, f'TR_forward_attn_best.png'))
                plt.close()

                vloss_best_index = i
                vloss_best = mse_loss
            if i in range(100):
                # Forward, TL
                plt.figure()
                plt1, = plt.plot(lamda, vlabels[0, 0:301].cpu(), label='Real')
                plt2, = plt.plot(lamda, voutput[0, 0:301].cpu(), label=f'Predict, MSE = {mse_loss * 100:.2f}%')
                plt.legend()
                plt.xlabel('lambda(nm)')
                plt.ylabel('TL')
                plt.title('Forward')

                if os.path.exists(os.path.join(figs_save_path, f'TL_forward_{i}.png')):
                    os.remove(os.path.join(figs_save_path, f'TL_forward_{i}.png'))
                plt.savefig(os.path.join(figs_save_path, f'TL_forward_{i}.png'), dpi=900)
                plt.close()

                # Forward, TR
                plt.figure()
                plt1, = plt.plot(lamda, vlabels[0, 301:].cpu(), label='Real')
                plt2, = plt.plot(lamda, voutput[0, 301:].cpu(), label=f'Predict, MSE = {mse_loss * 100:.2f}%')
                plt.legend()
                plt.xlabel('lambda(nm)')
                plt.ylabel('TR')
                plt.title('Forward')

                if os.path.exists(os.path.join(figs_save_path, f'TR_forward_{i}.png')):
                    os.remove(os.path.join(figs_save_path, f'TR_forward_{i}.png'))
                plt.savefig(os.path.join(figs_save_path, f'TR_forward_{i}.png'), dpi=900)
                plt.close()

        forward_mse_loss_sum = forward_mse_loss_sum / (i + 1)
        forward_mae_loss_sum = forward_mae_loss_sum / (i + 1)
        print(f'Forward MSE = {forward_mse_loss_sum}, MAE = {forward_mae_loss_sum}')

    # print(forward_model.self_attention.weight)
    attn_matrix = forward_model.self_attention.weight.to('cpu')
    results_save = {'attn_matrix': attn_matrix.tolist(), 'real': real, 'prediction': prediction,
                    'mse': forward_mse_loss_sum, 'mae': forward_mae_loss_sum}
    scio.savemat(os.path.join(RESULTS_PATH, timestamp, 'results.mat'), mdict=results_save)
