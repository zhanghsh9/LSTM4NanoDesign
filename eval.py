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
    EPOCHS, NUM_LAYERS

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    time_now = int(time.strftime("%Y%m%d%H%M%S", time.localtime()))
    torch.manual_seed(time_now)

    print('Running on {}'.format(device))
    print()

    # dir
    timestamp = '20230422'
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
    model_name = 'Forward_mse_vloss_best.pth'
    forward_model = torch.load(os.path.join(model_save_path, model_name))
    forward_model.to(device)

    loss_fn = MSELoss()
    mse_loss_sum = 0.
    with torch.no_grad():
        forward_model.eval()
        lamda = range(900, 1801, 3 * SAMPLE_RATE)

        for i, data in enumerate(forward_test_dataloader):
            vinputs, vlabels = data
            vinputs, vlabels = vinputs.float().to(device), vlabels.float().to(device)
            voutputs, _ = forward_model(vinputs)

            for j in range(len(vlabels)):
                plt1, = plt.plot(lamda, vlabels[0, :].cpu(), label='Real')
                plt2, = plt.plot(lamda, voutputs[0, :].cpu(), label='Prediction')
                mse_loss = loss_fn(vlabels, voutputs).item()
                mse_loss_sum += mse_loss
                plt.legend()
                plt.xlabel('lambda(nm)')
                plt.ylabel('TL')
                plt.title('Forward, mse = {}'.format(mse_loss))

                if os.path.exists(os.path.join(figs_save_path, 'forward_{}.png'.format(i))):
                    os.remove(os.path.join(figs_save_path, 'forward_{}.png'.format(i)))
                plt.savefig(os.path.join(figs_save_path, 'forward_{}.png'.format(i)))
                plt.show()

    print('MSE = {}'.format(mse_loss_sum / (i + 1)))
