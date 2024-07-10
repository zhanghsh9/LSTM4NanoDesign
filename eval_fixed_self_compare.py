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

from parameters import RESULTS_PATH, DATA_PATH, RODS, NUM_WORKERS, SAMPLE_RATE

if __name__ == '__main__':
    # Get device
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    else:
        device = torch.device('cuda:3')
        print(f'Running on {device} version = {torch.version.cuda}, device count = {torch.cuda.device_count()}')
        print()

    time_now = int(time.strftime("%Y%m%d%H%M%S", time.localtime()))
    torch.manual_seed(time_now)

    # Create dataloader
    transform = transforms.Compose([transforms.ToTensor()])
    _, test_dataset = create_dataset(data_path=DATA_PATH, rods=RODS, use_TL_TR=True, transform=transform,
                                     sample_rate=SAMPLE_RATE, make_spectrum_int=False, device=device)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, drop_last=True,
                                 pin_memory=True)
    forward_loss_fn = MSELoss()
    lamda = range(900, 1801, 3 * SAMPLE_RATE)

    from results.compare.self.parameters import HIDDEN_UNITS, NUM_LAYERS, ACTIVATE_FUNC

    # Load self attention model
    self_attention_model = torch.load(r'results/compare/self/models/Forward_mse_vloss_best_attn_0.pth')
    self_attention_model.to(device)
    self_attention_model.eval()
    self_prediction = []
    real = []

    from results.compare.fixed.parameters import HIDDEN_UNITS, NUM_LAYERS, ACTIVATE_FUNC

    # Load self attention model
    fixed_attention_model = torch.load(r'results/compare/fixed/models/Forward_mse_vloss_best_attn_16.5.pth')
    fixed_attention_model.to(device)
    fixed_attention_model.eval()
    fixed_prediction = []

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            vinputs, vlabels = data
            vinputs, vlabels = vinputs.float().to(device), vlabels.float().to(device)
            real.append(vlabels.to('cpu').tolist())
            self_voutput, _ = self_attention_model(vinputs)
            self_mse_loss = forward_loss_fn(vlabels, self_voutput).item()
            self_prediction.append(self_voutput.to('cpu').tolist())
            fixed_voutput, _ = fixed_attention_model(vinputs)
            fixed_mse_loss = forward_loss_fn(vlabels, fixed_voutput).item()
            fixed_prediction.append(fixed_voutput.to('cpu').tolist())

            # Forward, TL
            plt.figure()
            plt1, = plt.plot(lamda, vlabels[0, 0:301].cpu(), label='Real')
            plt2, = plt.plot(lamda, self_voutput[0, 0:301].cpu(), label=f'Self Attn, MSE = {self_mse_loss * 100:.2f}%')
            plt3, = plt.plot(lamda, fixed_voutput[0, 0:301].cpu(), label=f'Fixed Attn, MSE = {fixed_mse_loss * 100:.2f}%')
            plt.legend()
            plt.xlabel('lambda(nm)')
            plt.ylabel('TL')
            plt.title('Forward')

            if os.path.exists(os.path.join('results', 'compare', 'figs', f'TL_forward_{i}.png')):
                os.remove(os.path.join('results', 'compare', 'figs', f'TL_forward_{i}.png'))
            plt.savefig(os.path.join('results', 'compare', 'figs', f'TL_forward_{i}.png'), dpi=900)
            plt.show()
            plt.close()

            # Forward, TR
            plt.figure()
            plt1, = plt.plot(lamda, vlabels[0, 301:].cpu(), label='Real')
            plt2, = plt.plot(lamda, self_voutput[0, 301:].cpu(), label=f'Self Attn, MSE = {self_mse_loss * 100:.2f}%')
            plt3, = plt.plot(lamda, fixed_voutput[0, 301:].cpu(), label=f'Fixed Attn, MSE = {fixed_mse_loss * 100:.2f}%')
            plt.legend()
            plt.xlabel('lambda(nm)')
            plt.ylabel('TR')
            plt.title('Forward')

            if os.path.exists(os.path.join('results', 'compare', 'figs', f'TR_forward_{i}.png')):
                os.remove(os.path.join('results', 'compare', 'figs', f'TR_forward_{i}.png'))
            plt.savefig(os.path.join('results', 'compare', 'figs', f'TR_forward_{i}.png'), dpi=900)
            plt.show()
            plt.close()
