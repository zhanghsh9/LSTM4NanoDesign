import torch
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
import os
import time

from data import create_dataset

from parameters import RESULTS_PATH, DATA_PATH, FIGS_PATH, MODEL_PATH, RODS, BATCH_SIZE, NUM_WORKERS, SAMPLE_RATE, \
    EPOCHS, NUM_LAYERS, NUM_LSTMS, HIDDEN_UNITS

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    time_now = int(time.strftime("%Y%m%d%H%M%S", time.localtime()))
    torch.manual_seed(time_now)

    print('Running on {}'.format(device))
    print()

    # dir
    timestamp = '20230423'
    model_save_path = os.path.join(RESULTS_PATH, timestamp, MODEL_PATH)
    figs_save_path = os.path.join(RESULTS_PATH, timestamp, FIGS_PATH)

    # Create dataloader
    transform = transforms.Compose([transforms.ToTensor()])
    _, forward_test_dataset = create_dataset(data_path=DATA_PATH, rods=RODS, reverse=False, use_TL=True,
                                             transform=transform, sample_rate=SAMPLE_RATE)
    forward_test_dataloader = DataLoader(forward_test_dataset, batch_size=1, shuffle=False,
                                         num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)
    # Forward prediction
    '''
    model_name = 'Transformer_epochs_{}_dmodel_{}_dff_{}_heads_{}_layers_{}.pth'.format(EPOCHS, DIM_MODEL,
                                                                                        DIM_FEEDFORWARD, HEADS,
                                                                                        NUM_LAYERS)
    '''

    attn_list = [1, 3, 5, 7, 9]
    forward_model = []
    loss_fn = MSELoss()
    mse_loss_sum = [0 for _ in range(len(attn_list))]
    lamda = range(900, 1801, 3 * SAMPLE_RATE)

    for ii in range(len(attn_list)):
        model_name = 'Forward_epochs_{}_lstms_{}_hidden_{}_attn_{}.pth'.format(EPOCHS, NUM_LSTMS, HIDDEN_UNITS,
                                                                           attn_list[ii])
        forward_model.append(torch.load(os.path.join(model_save_path, model_name)))
        forward_model[ii].to(device)
        forward_model[ii].eval()

    with torch.no_grad():
        for i, data in enumerate(forward_test_dataloader):
            vinputs, vlabels = data
            vinputs, vlabels = vinputs.float().to(device), vlabels.float().to(device)
            voutputs=[]
            for ii in range(len(attn_list)):
                voutput, _ = forward_model[ii](vinputs)
                voutputs.append(voutput)

            for j in range(len(vlabels)):
                plt1, = plt.plot(lamda, vlabels[0, :].cpu(), label='Real')
                for ii in range(len(attn_list)):
                    plt2, = plt.plot(lamda, voutputs[ii][0, :].cpu(), label='Prediction_attn_{}'.format(attn_list[ii]))
                    mse_loss = loss_fn(vlabels, voutputs[ii]).item()
                    mse_loss_sum[ii] = mse_loss_sum[ii] + mse_loss

                plt.legend()
                plt.xlabel('lambda(nm)')
                plt.ylabel('TL')
                plt.title('Forward')

                if os.path.exists(os.path.join(figs_save_path, 'forward_{}.png'.format(i))):
                    os.remove(os.path.join(figs_save_path, 'forward_{}.png'.format(i)))
                plt.savefig(os.path.join(figs_save_path, 'forward_{}.png'.format(i)))
                plt.show()

    for ii in range(len(attn_list)):
        print('Attention = {}, MSE = {}'.format(attn_list[ii], mse_loss_sum[ii] / (i + 1)))

    plt1, = plt.plot(attn_list, mse_loss_sum, label='MSE')
    plt.legend()
    plt.xlabel('Attention')
    plt.ylabel('MSE')
    plt.title('MSE to attention')
    if os.path.exists(os.path.join(figs_save_path, 'MST_Attn.png')):
        os.remove(os.path.join(figs_save_path, 'MST_Attn.png'))
    plt.savefig(os.path.join(figs_save_path, 'MST_Attn.png'))
    plt.show()

