import torch
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import scipy.io as scio

from data import create_dataset

from parameters import RESULTS_PATH, DATA_PATH, FIGS_PATH, MODEL_PATH, RODS, BATCH_SIZE, NUM_WORKERS, SAMPLE_RATE, \
    EPOCHS, NUM_LAYERS, HIDDEN_UNITS

if __name__ == '__main__':
    # Get device
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    else:
        device = torch.device('cuda:0')
        print(f'Running on {device} version = {torch.version.cuda}, device count = {torch.cuda.device_count()}')
        print()

    time_now = int(time.strftime("%Y%m%d%H%M%S", time.localtime()))
    torch.manual_seed(time_now)

    # dir
    timestamp = '20240806_tanh'
    RESULTS_PATH = os.path.join(RESULTS_PATH, 'fixed_attention')
    model_save_path = os.path.join(RESULTS_PATH, timestamp, MODEL_PATH)
    figs_save_path = os.path.join(RESULTS_PATH, timestamp, FIGS_PATH)

    # Create dataloader
    transform = transforms.Compose([transforms.ToTensor()])
    _, test_dataset = create_dataset(data_path=DATA_PATH, rods=RODS, use_TL_TR=True, transform=transform,
                                     sample_rate=SAMPLE_RATE, make_spectrum_int=False, device=device)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, drop_last=True,
                                 pin_memory=True)
    # Forward prediction
    '''
    model_name = 'Transformer_epochs_{}_dmodel_{}_dff_{}_heads_{}_layers_{}.pth'.format(EPOCHS, DIM_MODEL,
                                                                                        DIM_FEEDFORWARD, HEADS,
                                                                                        NUM_LAYERS)
    '''

    # attn_list = [i / 2. for i in range(12)]
    attn_list = np.arange(0.5, 20.5, 1).tolist()
    attn_list.append(1)
    attn_list.sort()
    forward_model = []
    forward_loss_fn = MSELoss()
    forward_mse_loss_sum = [0] * len(attn_list)
    lamda = range(900, 1801, 3 * SAMPLE_RATE)
    prediction = []
    real = []

    # Load models
    for ii in range(len(attn_list)):
        # Forward model
        # model_name = f'Forward_epochs_{EPOCHS}_lstms_{len(HIDDEN_UNITS)}_hidden_{HIDDEN_UNITS}_attn_{attn_list[ii]}.pth'
        model_name = f'Forward_mse_vloss_best_attn_{attn_list[ii]}.pth'
        forward_model = torch.load(os.path.join(model_save_path, model_name))
        forward_model.to(device)
        forward_model.eval()
        prediction.append([])
        real.append([])
        vloss_best = 100
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                vinputs, vlabels = data
                vinputs, vlabels = vinputs.float().to(device), vlabels.float().to(device)
                voutput, _ = forward_model(vinputs)
                mse_loss = forward_loss_fn(vlabels, voutput).item()
                forward_mse_loss_sum[ii] = forward_mse_loss_sum[ii] + mse_loss
                prediction[ii].append(voutput.to('cpu').tolist())
                real[ii].append(vlabels.to('cpu').tolist())

                if mse_loss < vloss_best:
                    plt.figure()
                    plt1, = plt.plot(lamda, vlabels[0, 0:301].cpu(), label='Real')
                    plt2, = plt.plot(lamda, voutput[0, 0:301].cpu(), label=f'Attn_{attn_list[ii]}, MSE = {mse_loss*100:.2f}%')
                    plt.legend()
                    plt.xlabel('lambda(nm)')
                    plt.ylabel('TL')
                    plt.title('Forward')

                    if os.path.exists(os.path.join(figs_save_path, f'TL_forward_attn_{attn_list[ii]}_best.png')):
                        os.remove(os.path.join(figs_save_path, f'TL_forward_attn_{attn_list[ii]}_best.png'))
                    plt.savefig(os.path.join(figs_save_path, f'TL_forward_attn_{attn_list[ii]}_best.png'))
                    plt.close()

                    # Forward, TR
                    plt.figure()
                    plt1, = plt.plot(lamda, vlabels[0, 301:].cpu(), label='Real')
                    plt2, = plt.plot(lamda, voutput[0, 301:].cpu(), label=f'Attn_{attn_list[ii]}, MSE = {mse_loss*100:.2f}%')
                    plt.legend()
                    plt.xlabel('lambda(nm)')
                    plt.ylabel('TR')
                    plt.title('Forward')

                    if os.path.exists(os.path.join(figs_save_path, f'TR_forward_attn_{attn_list[ii]}_best.png')):
                        os.remove(os.path.join(figs_save_path, f'TR_forward_attn_{attn_list[ii]}_best.png'))
                    plt.savefig(os.path.join(figs_save_path, f'TR_forward_attn_{attn_list[ii]}_best.png'))
                    plt.close()
                    vloss_best_index = i
                    vloss_best = mse_loss

                if i in range(20):
                    # Forward, TL
                    plt.figure()
                    plt1, = plt.plot(lamda, vlabels[0, 0:301].cpu(), label='Real')
                    plt2, = plt.plot(lamda, voutput[0, 0:301].cpu(), label=f'Attn_{attn_list[ii]}, MSE = {mse_loss*100:.2f}%')
                    plt.legend()
                    plt.xlabel('lambda(nm)')
                    plt.ylabel('TL')
                    plt.title('Forward')

                    if os.path.exists(os.path.join(figs_save_path, f'TL_forward_{i}_attn_{attn_list[ii]}.png')):
                        os.remove(os.path.join(figs_save_path, f'TL_forward_{i}_attn_{attn_list[ii]}.png'))
                    plt.savefig(os.path.join(figs_save_path, f'TL_forward_{i}_attn_{attn_list[ii]}.png'))
                    plt.close()

                    # Forward, TR
                    plt.figure()
                    plt1, = plt.plot(lamda, vlabels[0, 301:].cpu(), label='Real')
                    plt2, = plt.plot(lamda, voutput[0, 301:].cpu(), label=f'Attn_{attn_list[ii]}, MSE = {mse_loss*100:.2f}%')
                    plt.legend()
                    plt.xlabel('lambda(nm)')
                    plt.ylabel('TR')
                    plt.title('Forward')

                    if os.path.exists(os.path.join(figs_save_path, f'TR_forward_{i}_attn_{attn_list[ii]}.png')):
                        os.remove(os.path.join(figs_save_path, f'TR_forward_{i}_attn_{attn_list[ii]}.png'))
                    plt.savefig(os.path.join(figs_save_path, f'TR_forward_{i}_attn_{attn_list[ii]}.png'))
                    plt.close()

    for ii in range(len(attn_list)):
        forward_mse_loss_sum[ii] = forward_mse_loss_sum[ii] / (i + 1)
        print(f'Attention = {attn_list[ii]}, forward MSE = {forward_mse_loss_sum[ii]}')

    plt1, = plt.plot(attn_list, forward_mse_loss_sum, label='forward')
    plt.legend()
    plt.xlabel('Attention')
    plt.ylabel('MSE')
    plt.title('MSE to attention')
    if os.path.exists(os.path.join(figs_save_path, 'MSE_Attn.png')):
        os.remove(os.path.join(figs_save_path, 'MSE_Attn.png'))
    plt.savefig(os.path.join(figs_save_path, 'MSE_Attn.png'))
    plt.show()

    results_save = {'attn_list': attn_list, 'forward_mse_loss_sum': forward_mse_loss_sum, 'real': real,
                    'prediction': prediction}
    scio.savemat(os.path.join(RESULTS_PATH, timestamp, 'results.mat'), mdict=results_save)

    print(f'Loss_min = {min(forward_mse_loss_sum)}, when attention = {attn_list[forward_mse_loss_sum.index(min(forward_mse_loss_sum))]}')
