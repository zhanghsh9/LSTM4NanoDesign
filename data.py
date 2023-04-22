import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import scipy.io as scio
import os
from math import sqrt

from parameters import DATA_PATH, RODS, SAMPLE_RATE


def get_filename(path, rods):
    """
    Get training data filename, .mat
    :param path: str
    :param rods: int
    :return: filename
    """
    for root, dirs, files in os.walk(path):
        for name in files:
            if int(name[5]) == rods:
                if name[12:-4] == 'test':
                    test_filename = name
                elif name[12:-4] == 'train':
                    train_filename = name
    return train_filename, test_filename


def entire_normalize(data):
    data_mean = 0
    data_var = 0
    for i in data:
        data_mean += i
        data_var += i * i
    data_mean = data_mean / (len(data))
    data_var = data_var / (len(data))
    normalized_data = []
    for i in data:
        temp = (i - data_mean) / sqrt(data_var)
        normalized_data.append(temp)
    return normalized_data, data_mean, data_var


def create_dataset(data_path=DATA_PATH, rods=RODS, reverse=False, use_TL=True, transform=None, sample_rate=SAMPLE_RATE):
    """
    Create dataset
    :param data_path: str
    :param rods: int or str
    :param reverse: bool
    :param use_TL: bool
    :param transform: torchvision.transforms
    :param sample_rate: int
    :return: dataset
    """

    train_filename, test_filename = get_filename(data_path, rods)

    train_data_path = os.path.join(DATA_PATH, train_filename)
    test_data_path = os.path.join(DATA_PATH, test_filename)
    print('Using {} as training data'.format(train_data_path))
    print('Using {} as test data'.format(test_data_path))
    print()

    train_dataset = GoldNanorodSingle(data_path=train_data_path, reverse=reverse, use_TL=use_TL,
                                      transform=transform, sample_rate=sample_rate)
    test_dataset = GoldNanorodSingle(data_path=test_data_path, reverse=reverse, use_TL=use_TL, transform=transform,
                                     sample_rate=sample_rate)

    return train_dataset, test_dataset


class GoldNanorodSingle(Dataset):
    def __init__(self, data_path=DATA_PATH, rods=RODS, reverse=False, use_TL=True, transform=None,
                 sample_rate=SAMPLE_RATE):
        data = scio.loadmat(data_path)

        # Parameters
        self.data_path = data_path
        self.rods = rods
        self.reverse = reverse
        self.use_TL = use_TL
        self.transform = transform
        self.sample_rate = sample_rate

        x = [[] for j in range(int(len(data['normal00'][0]) / 6))]
        y = [[] for j in range(int(len(data['normal00'][0]) / 6))]
        z = [[] for j in range(int(len(data['normal00'][0]) / 6))]
        l = [[] for j in range(int(len(data['normal00'][0]) / 6))]
        r = [[] for j in range(int(len(data['normal00'][0]) / 6))]
        t = [[] for j in range(int(len(data['normal00'][0]) / 6))]

        self.norm_x = [[] for j in range(int(len(data['normal00'][0]) / 6))]
        self.norm_y = [[] for j in range(int(len(data['normal00'][0]) / 6))]
        self.norm_z = [[] for j in range(int(len(data['normal00'][0]) / 6))]
        self.norm_l = [[] for j in range(int(len(data['normal00'][0]) / 6))]
        self.norm_t = [[] for j in range(int(len(data['normal00'][0]) / 6))]

        self.x_mean = [0 for j in range(int(len(data['normal00'][0]) / 6))]
        self.x_var = [0 for j in range(int(len(data['normal00'][0]) / 6))]
        self.y_mean = [0 for j in range(int(len(data['normal00'][0]) / 6))]
        self.y_var = [0 for j in range(int(len(data['normal00'][0]) / 6))]
        self.z_mean = [0 for j in range(int(len(data['normal00'][0]) / 6))]
        self.z_var = [0 for j in range(int(len(data['normal00'][0]) / 6))]
        self.l_mean = [0 for j in range(int(len(data['normal00'][0]) / 6))]
        self.l_var = [0 for j in range(int(len(data['normal00'][0]) / 6))]
        self.t_mean = [0 for j in range(int(len(data['normal00'][0]) / 6))]
        self.t_var = [0 for j in range(int(len(data['normal00'][0]) / 6))]

        # De-package
        for i in range(len(data['normal00'])):
            for j in range(int(len(data['normal00'][i]) / 6)):
                x[j].append(int(data['normal00'][i][6 * j]))
                y[j].append(int(data['normal00'][i][6 * j + 1]))
                z[j].append(int(data['normal00'][i][6 * j + 2]))
                l[j].append(int(data['normal00'][i][6 * j + 3]))
                r[j].append(data['normal00'][i][6 * j + 3] / data['normal00'][i][6 * j + 4])
                t[j].append(int(data['normal00'][i][6 * j + 5]))

        # Normalize
        for i in range(len(x)):
            self.norm_x[i], self.x_mean[i], self.x_var[i] = entire_normalize(x[i])
            self.norm_y[i], self.y_mean[i], self.y_var[i] = entire_normalize(y[i])
            self.norm_z[i], self.z_mean[i], self.z_var[i] = entire_normalize(z[i])
            self.norm_l[i], self.l_mean[i], self.l_var[i] = entire_normalize(l[i])
            self.norm_t[i], self.t_mean[i], self.t_var[i] = entire_normalize(t[i])

        normal00 = []
        for i in range(len(data['normal00'])):
            temp = []
            for j in range(int(len(data['normal00'][i]) / 6)):
                temp.extend(
                    [self.norm_x[j][i], self.norm_y[j][i], self.norm_z[j][i], self.norm_l[j][i],
                     self.norm_l[j][i] / r[j][i], self.norm_t[j][i]])
            normal00.append(temp)
        normal00 = torch.Tensor(normal00)
        if reverse:
            if use_TL:
                self.normal00 = [l[::sample_rate] for l in data['TL']]
            else:
                self.normal00 = [l[::sample_rate] for l in data['TR']]
            self.spectra = normal00
        else:
            if use_TL:
                self.spectra = [l[::sample_rate] for l in data['TL']]
            else:
                self.spectra = [l[::sample_rate] for l in data['TR']]
            self.normal00 = normal00

        # Get seq len
        self.src_len = torch.Tensor([len(l) for l in self.normal00]).to(torch.int32)
        self.tgt_len = torch.Tensor([len(l) for l in self.spectra]).to(torch.int32)
        self.max_src_seq_len = int(max(self.src_len))
        self.max_tgt_seq_len = int(max(self.tgt_len))

        self.normal00 = torch.Tensor(self.normal00).to(torch.int32)
        self.spectra = torch.Tensor(self.spectra).to(torch.int32)

    def __len__(self):
        return len(self.normal00)

    def __getitem__(self, index):
        paras = self.normal00[index]
        result = self.spectra[index]
        return paras, result

    def print(self):
        print('class GoldNanorodSingle with: ')
        print('data path = {}'.format(self.data_path))
        print('rods = {}'.format(self.rods))
        print('reverse = {}'.format(self.reverse))
        print('use TL = {}'.format(self.use_TL))
        print('transform = {}'.format(self.transform))
        print('spectrum sample rate = {}'.format(self.sample_rate))
        print('source size = [{}, {}]'.format(len(self.normal00), self.max_src_seq_len))
        print('target size = [{}, {}]'.format(len(self.spectra), self.max_tgt_seq_len))

    def print_item(self, index):
        print('source = {}'.format(self.normal00[index]))
        print('target = {}'.format(self.spectra[index]))
