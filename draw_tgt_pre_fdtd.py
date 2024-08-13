import torch
from torch.nn import MSELoss
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import scipy.io as scio

from data import create_dataset

from parameters import RESULTS_PATH, DATA_PATH, FIGS_PATH, MODEL_PATH, RODS, NUM_WORKERS, SAMPLE_RATE

if __name__ == '__main__':
    # dir
    timestamp = '20240714_leakyrelu'
    RESULTS_PATH = os.path.join(RESULTS_PATH, 'backwards/self_attention')
    model_save_path = os.path.join(RESULTS_PATH, timestamp, MODEL_PATH)
    figs_save_path = os.path.join(RESULTS_PATH, timestamp, FIGS_PATH)

    lamda = range(900, 1801, 3 * SAMPLE_RATE)
    prediction_data = scio.loadmat(os.path.join(RESULTS_PATH, timestamp, 'results.mat'))
    tandem_output = prediction_data['tandem_output']
    target = prediction_data['target']
    fdtd_data = scio.loadmat(os.path.join(RESULTS_PATH, timestamp, 'spectrum.mat'))
    fdtd_results = fdtd_data['TL_TR']
    vloss_best = 100
    vloss_best_index = 0
    vloss_best_tandem = 100
    loss_fn = MSELoss()
    fdtd_mse_rec = []

    for i in range(len(target)):
        mse_loss_tandem = loss_fn(torch.Tensor(target[i]), torch.Tensor(tandem_output[i])).item()
        mse_loss_fdtd = loss_fn(torch.Tensor(target[i]), torch.Tensor(fdtd_results[i])).item()
        fdtd_mse_rec.append(mse_loss_fdtd)

        if mse_loss_fdtd < vloss_best:
            vloss_best_index = i
            vloss_best = mse_loss_fdtd
            vloss_best_tandem = mse_loss_tandem
        '''
        plt.figure()
        plt1, = plt.plot(lamda, target[i, 0:301], label='Target')
        plt2, = plt.plot(lamda, tandem_output[i, 0:301],
                         label=f'Tandem output, MSE = {mse_loss_tandem * 100:.2f}%')
        plt3, = plt.plot(lamda, fdtd_results[i, 0:301],
                         label=f'FDTD simulation, MSE = {mse_loss_fdtd * 100:.2f}%')
        plt.legend()
        plt.xlabel('lambda(nm)')
        plt.ylabel('TL')
        plt.title('Backward')

        if os.path.exists(os.path.join(figs_save_path, f'TL_backward_{i}.png')):
            os.remove(os.path.join(figs_save_path, f'TL_backward_{i}.png'))
        plt.savefig(os.path.join(figs_save_path, f'TL_backward_{i}.png'), dpi=900)
        # plt.show()
        plt.close()

        plt.figure()
        plt1, = plt.plot(lamda, target[i, 301:], label='Target')
        plt2, = plt.plot(lamda, tandem_output[i, 301:],
                         label=f'Tandem output, MSE = {mse_loss_tandem * 100:.2f}%')
        plt3, = plt.plot(lamda, fdtd_results[i, 301:],
                         label=f'FDTD simulation, MSE = {mse_loss_fdtd * 100:.2f}%')
        plt.legend()
        plt.xlabel('lambda(nm)')
        plt.ylabel('TR')
        plt.title('Backward')

        if os.path.exists(os.path.join(figs_save_path, f'TR_backward_{i}.png')):
            os.remove(os.path.join(figs_save_path, f'TR_backward_{i}.png'))
        plt.savefig(os.path.join(figs_save_path, f'TR_backward_{i}.png'), dpi=900)
        # plt.show()
        plt.close()

    fdtd_mse_sum = fdtd_mse_sum / (i + 1)
    print(f'Backward FDTD MSE = {fdtd_mse_sum}')
    
    plt.figure()
    plt1, = plt.plot(lamda, target[vloss_best_index, 0:301], label='Target')
    plt2, = plt.plot(lamda, tandem_output[vloss_best_index, 0:301],
                     label=f'Tandem output, MSE = {vloss_best_tandem * 100:.2f}%')
    plt3, = plt.plot(lamda, fdtd_results[vloss_best_index, 0:301],
                     label=f'FDTD simulation, MSE = {vloss_best * 100:.2f}%')
    plt.legend()
    plt.xlabel('lambda(nm)')
    plt.ylabel('TL')
    plt.title('Backward')

    if os.path.exists(os.path.join(figs_save_path, f'TL_backward_best.png')):
        os.remove(os.path.join(figs_save_path, f'TL_backward_best.png'))
    plt.savefig(os.path.join(figs_save_path, f'TL_backward_best.png'), dpi=900)
    # plt.show()
    plt.close()

    plt.figure()
    plt1, = plt.plot(lamda, target[vloss_best_index, 301:], label='Target')
    plt2, = plt.plot(lamda, tandem_output[vloss_best_index, 301:],
                     label=f'Tandem output, MSE = {vloss_best_tandem * 100:.2f}%')
    plt3, = plt.plot(lamda, fdtd_results[vloss_best_index, 301:],
                     label=f'FDTD simulation, MSE = {vloss_best * 100:.2f}%')
    plt.legend()
    plt.xlabel('lambda(nm)')
    plt.ylabel('TR')
    plt.title('Backward')

    if os.path.exists(os.path.join(figs_save_path, f'TR_backward_best.png')):
        os.remove(os.path.join(figs_save_path, f'TR_backward_best.png'))
    plt.savefig(os.path.join(figs_save_path, f'TR_backward_best.png'), dpi=900)
    # plt.show()
    plt.close()
    '''
    scio.savemat(os.path.join(RESULTS_PATH, timestamp, 'fdtd_mes_rec.mat'), {'fdtd_mes_rec': fdtd_mse_rec})
