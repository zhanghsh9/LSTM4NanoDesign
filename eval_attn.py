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
# device = 'cpu'

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
    _, test_dataset = create_dataset(data_path=DATA_PATH, rods=RODS, use_TL=True, transform=transform,
                                     sample_rate=SAMPLE_RATE)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, drop_last=True,
                                 pin_memory=True)
    # Forward prediction
    '''
    model_name = 'Transformer_epochs_{}_dmodel_{}_dff_{}_heads_{}_layers_{}.pth'.format(EPOCHS, DIM_MODEL,
                                                                                        DIM_FEEDFORWARD, HEADS,
                                                                                        NUM_LAYERS)
    '''

    attn_list = [0., 1., 2., 3., 4.]
    forward_model = []
    forward_loss_fn = MSELoss()
    forward_mse_loss_sum = [0 for _ in range(len(attn_list))]
    backward_model = []
    backward_loss_fn = MSELoss()
    backward_mse_loss_sum = [0 for _ in range(len(attn_list))]
    lamda = range(900, 1801, 3 * SAMPLE_RATE)

    for ii in range(len(attn_list)):
        # Forward model
        model_name = 'Forward_mse_vloss_best_attn_{}.pth'.format(attn_list[ii])
        forward_model.append(torch.load(os.path.join(model_save_path, model_name)))
        forward_model[ii].to(device)
        forward_model[ii].eval()

        # Backward model
        model_name = 'Backward_mse_vloss_best_attn_{}.pth'.format(attn_list[ii])
        backward_model.append(torch.load(os.path.join(model_save_path, model_name)))
        backward_model[ii].to(device)
        backward_model[ii].eval()

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            vinputs, vlabels = data
            vinputs, vlabels = vinputs.float().to(device), vlabels.float().to(device)

            # Forward
            forward_voutputs = []
            for ii in range(len(attn_list)):
                voutput, _ = forward_model[ii](vinputs)
                forward_voutputs.append(voutput)

            for j in range(len(vlabels)):
                plt1, = plt.plot(lamda, vlabels[0, :].cpu(), label='Real')
                for ii in range(len(attn_list)):
                    plt2, = plt.plot(lamda, forward_voutputs[ii][0, :].cpu(), label='Attn_{}'.format(attn_list[ii]))
                    mse_loss = forward_loss_fn(vlabels, forward_voutputs[ii]).item()
                    forward_mse_loss_sum[ii] = forward_mse_loss_sum[ii] + mse_loss

                plt.legend()
                plt.xlabel('lambda(nm)')
                plt.ylabel('TL')
                plt.title('Forward')

                if os.path.exists(os.path.join(figs_save_path, 'forward_{}.png'.format(i))):
                    os.remove(os.path.join(figs_save_path, 'forward_{}.png'.format(i)))
                plt.savefig(os.path.join(figs_save_path, 'forward_{}.png'.format(i)))
                plt.show()

            # Backward
            backward_voutputs = []
            for ii in range(len(attn_list)):
                voutput, _ = backward_model[ii](vlabels)
                voutput, _ = forward_model[ii](voutput)
                backward_voutputs.append(voutput)

            for j in range(len(vlabels)):
                plt1, = plt.plot(lamda, vlabels[0, :].cpu(), label='Real')
                for ii in range(len(attn_list)):
                    plt2, = plt.plot(lamda, backward_voutputs[ii][0, :].cpu(), label='Attn_{}'.format(attn_list[ii]))
                    mse_loss = backward_loss_fn(vlabels, backward_voutputs[ii]).item()
                    backward_mse_loss_sum[ii] = backward_mse_loss_sum[ii] + mse_loss

                plt.legend()
                plt.xlabel('lambda(nm)')
                plt.ylabel('TL')
                plt.title('Backward')

                if os.path.exists(os.path.join(figs_save_path, 'backward_{}.png'.format(i))):
                    os.remove(os.path.join(figs_save_path, 'backward_{}.png'.format(i)))
                plt.savefig(os.path.join(figs_save_path, 'backward_{}.png'.format(i)))
                plt.show()


    for ii in range(len(attn_list)):
        forward_mse_loss_sum[ii] = forward_mse_loss_sum[ii] / (i + 1)
        backward_mse_loss_sum[ii] = backward_mse_loss_sum[ii] / (i + 1)
        print('Attention = {}, forward MSE = {}, backward MSE = {}'.format(attn_list[ii], forward_mse_loss_sum[ii],
                                                                           backward_mse_loss_sum[ii]))

    plt1, = plt.plot(attn_list, forward_mse_loss_sum, label='MSE')
    plt.legend()
    plt.xlabel('Attention')
    plt.ylabel('MSE')
    plt.title('MSE to attention')
    if os.path.exists(os.path.join(figs_save_path, 'MSE_Attn.png')):
        os.remove(os.path.join(figs_save_path, 'MSE_Attn.png'))
    plt.savefig(os.path.join(figs_save_path, 'MSE_Attn.png'))
    plt.show()
