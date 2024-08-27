import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np
import scipy.io as scio
import os
from math import sqrt
import time

from parameters import DATA_PATH, RODS, SAMPLE_RATE


def get_filename_delta(path, rods):
    """
    Get training data filename, .mat
    :param path: str
    :param rods: int
    :return: filename
    """

    for root, dirs, files in os.walk(path):
        for name in files:
            if name[13] == f'{rods}' and name.startswith('deltaed'):
                if name.endswith('test.mat'):
                    test_filename = name
                elif name.endswith('train.mat'):
                    train_filename = name
    return train_filename, test_filename


def get_filename(path, rods):
    """
    Get training data filename, .mat
    :param path: str
    :param rods: int
    :return: filename
    """

    for root, dirs, files in os.walk(path):
        for name in files:
            if name[5] == f'{rods}':
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


def create_dataset(data_path=DATA_PATH, rods=RODS, use_TL_TR="TL_TR", transform=None,
                   sample_rate=SAMPLE_RATE, make_spectrum_int=False, device=torch.device('cuda')):
    """
    Create dataset
    :param device:
    :param make_spectrum_int:
    :param data_path: str
    :param rods: int or str
    :param use_TL_TR: bool
    :param transform: torchvision.transforms
    :param sample_rate: int
    :return: dataset
    """

    train_filename, test_filename = get_filename(data_path, rods)

    train_data_path = os.path.join(DATA_PATH, train_filename)
    test_data_path = os.path.join(DATA_PATH, test_filename)
    print(f'{time.strftime("%Y%m%d  %H:%M:%S", time.localtime())}: Using {train_data_path} as training data')
    print(f'{time.strftime("%Y%m%d  %H:%M:%S", time.localtime())}: Using {test_data_path} as test data')
    print()

    train_dataset = GoldNanorodSingle(data_path=train_data_path, use_TL_TR=use_TL_TR, transform=transform,
                                      sample_rate=sample_rate, make_spectrum_int=make_spectrum_int, device=device)
    test_dataset = GoldNanorodSingle(data_path=test_data_path, use_TL_TR=use_TL_TR, transform=transform,
                                     sample_rate=sample_rate, make_spectrum_int=make_spectrum_int, device=device)

    return train_dataset, test_dataset


def create_dataset_delta(data_path=DATA_PATH, rods=RODS, use_TL_TR=True, transform=None,
                         sample_rate=SAMPLE_RATE, make_spectrum_int=False, device=torch.device('cuda')):
    """
    Create dataset
    :param device:
    :param make_spectrum_int:
    :param data_path: str
    :param rods: int or str
    :param use_TL_TR: bool
    :param transform: torchvision.transforms
    :param sample_rate: int
    :return: dataset
    """

    train_filename, test_filename = get_filename_delta(data_path, rods)

    train_data_path = os.path.join(DATA_PATH, train_filename)
    test_data_path = os.path.join(DATA_PATH, test_filename)
    print(f'{time.strftime("%Y%m%d  %H:%M:%S", time.localtime())}: Using {train_data_path} as training data')
    print(f'{time.strftime("%Y%m%d  %H:%M:%S", time.localtime())}: Using {test_data_path} as test data')
    print()

    train_dataset = GoldNanorodDelta(data_path=train_data_path, use_TL_TR=use_TL_TR, transform=transform,
                                     sample_rate=sample_rate, make_spectrum_int=make_spectrum_int, device=device)
    test_dataset = GoldNanorodDelta(data_path=test_data_path, use_TL_TR=use_TL_TR, transform=transform,
                                    sample_rate=sample_rate, make_spectrum_int=make_spectrum_int, device=device)

    return train_dataset, test_dataset


class GoldNanorodSingle(Dataset):
    def __init__(self, data_path=DATA_PATH, rods=RODS, use_TL_TR='TL_TR', transform=None,
                 sample_rate=SAMPLE_RATE, make_spectrum_int=False, device=torch.device('cuda')):
        data = scio.loadmat(data_path)

        # Parameters
        self.data_path = data_path
        self.rods = rods
        self.use_TL_TR = use_TL_TR
        self.transform = transform
        self.sample_rate = sample_rate
        self.make_spectrum_int = make_spectrum_int
        self.device = device

        # Process parameter vectors
        '''
        x = [[] for j in range(int(len(data['normal00'][0]) / 6))]
        y = [[] for j in range(int(len(data['normal00'][0]) / 6))]
        z = [[] for j in range(int(len(data['normal00'][0]) / 6))]
        l = [[] for j in range(int(len(data['normal00'][0]) / 6))]
        r = [[] for j in range(int(len(data['normal00'][0]) / 6))]
        t = [[] for j in range(int(len(data['normal00'][0]) / 6))]
        
        x = data['x']
        y = data['y']
        z = data['z']
        l = data['l']
        r = data['r']
        t = data['t']
        
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
        '''
        norm_normal00 = data['norm_normal00']
        self.norm_normal00 = torch.Tensor(norm_normal00)
        self.x_mean = data['x_mean']
        self.x_std = data['x_std']
        self.y_mean = data['y_mean']
        self.y_std = data['y_std']
        self.z_mean = data['z_mean']
        self.z_std = data['z_std']
        self.l_mean = data['l_mean']
        self.l_std = data['l_std']
        self.t_mean = data['t_mean']
        self.t_std = data['t_std']
        self.r = data['r']

        '''
        TL = [[0 for j in range(len(data['TL'][i]))] for i in range(len(data['TL']))]
        TR = [[0 for j in range(len(data['TR'][i]))] for i in range(len(data['TR']))]
        TL_TR = [[0 for j in range(len(data['TL_TR'][i]))] for i in range(len(data['TL_TR']))]
        '''

        if self.make_spectrum_int:
            TL = data['TL_int']
            TR = data['TR_int']
            TL_TR = data['TL_TR_int']
            CD = TL - TR
        else:
            TL = data['TL_float']
            TR = data['TR_float']
            TL_TR = data['TL_TR_float']

            CD = 10*(TL - TR)

            '''
            for i in range(len(data['TL'])):
                for j in range(len(data['TL'][i])):
                    TL[i][j] = data['TL'][i][j] / 1000.
                    TR[i][j] = data['TR'][i][j] / 1000.

            for i in range(len(data['TL_TR'])):
                for j in range(len(data['TL_TR'][i])):
                    TL_TR[i][j] = data['TL_TR'][i][j] / 1000.
            '''

        if self.use_TL_TR == 'TL_TR':
            self.spectra = [list(l[::sample_rate]) for l in TL_TR]
        elif self.use_TL_TR == 'TL':
            self.spectra = [list(l[::sample_rate]) for l in TL]
        elif self.use_TL_TR == 'TR':
            self.spectra = [list(l[::sample_rate]) for l in TR]
        elif self.use_TL_TR == 'CD':
            self.spectra = [list(l[::sample_rate]) for l in CD]
        else:
            raise 'Error spectrum type!'
        # self.norm_normal00 = norm_normal00

        # Get seq len
        self.src_len = torch.Tensor([len(l) for l in self.norm_normal00]).to(torch.int32)
        self.tgt_len = torch.Tensor([len(l) for l in self.spectra]).to(torch.int32)
        self.max_src_seq_len = int(max(self.src_len))
        self.max_tgt_seq_len = int(max(self.tgt_len))

        # self.norm_normal00 = torch.Tensor(self.norm_normal00)
        self.spectra = torch.Tensor(self.spectra)

    def __len__(self):
        return len(self.norm_normal00)

    def __getitem__(self, index):
        paras = self.norm_normal00[index]
        result = self.spectra[index]
        return paras, result

    def print(self):
        print('class GoldNanorodSingle with: ')
        print('data path = {}'.format(self.data_path))
        print('rods = {}'.format(self.rods))
        print('device = {}'.format(self.device))
        print('use TL_TR = {}'.format(self.use_TL_TR))
        print('make spectrum integer = {}'.format(self.make_spectrum_int))
        print('transform = {}'.format(self.transform))
        print('spectrum sample rate = {}'.format(self.sample_rate))
        print('source size = [{}, {}]'.format(len(self.norm_normal00), self.max_src_seq_len))
        print('target size = [{}, {}]'.format(len(self.spectra), self.max_tgt_seq_len))

    def print_item(self, index):
        print('source = {}'.format(self.norm_normal00[index]))
        print('target = {}'.format(self.spectra[index]))


class GoldNanorodDelta(Dataset):
    def __init__(self, data_path=DATA_PATH, rods=RODS, use_TL_TR='TL_TR', transform=None,
                 sample_rate=SAMPLE_RATE, make_spectrum_int=False, device=torch.device('cuda')):
        data = scio.loadmat(data_path)

        # Parameters
        self.data_path = data_path
        self.rods = rods
        self.use_TL_TR = use_TL_TR
        self.transform = transform
        self.sample_rate = sample_rate
        self.make_spectrum_int = make_spectrum_int
        self.device = device

        norm_normal00 = data['norm_normal00']
        self.norm_normal00 = torch.Tensor(norm_normal00)
        self.delta_x_mean = data['delta_x_mean']
        self.delta_x_std = data['delta_x_std']
        self.delta_y_mean = data['delta_y_mean']
        self.delta_y_std = data['delta_y_std']
        self.delta_z_mean = data['delta_z_mean']
        self.delta_z_std = data['delta_z_std']
        self.l_mean = data['l_mean']
        self.l_std = data['l_std']
        self.t_mean = data['t_mean']
        self.t_std = data['t_std']
        self.r = data['r']

        if self.make_spectrum_int:
            TL = data['TL_int']
            TR = data['TR_int']
            TL_TR = data['TL_TR_int']
        else:
            TL = data['TL_float']
            TR = data['TR_float']
            TL_TR = data['TL_TR_float']

        if self.use_TL_TR == 'TL_TR':
            self.spectra = [list(l[::sample_rate]) for l in TL_TR]
        elif self.use_TL_TR == 'TL':
            self.spectra = [list(l[::sample_rate]) for l in TL]
        elif self.use_TL_TR == 'TR':
            self.spectra = [list(l[::sample_rate]) for l in TR]
        elif self.use_TL_TR == 'CD':
            self.spectra = [list(l[::sample_rate]) for l in CD]
        else:
            raise 'Error spectrum type!'

        # Get seq len
        self.src_len = torch.Tensor([len(l) for l in self.norm_normal00]).to(torch.int32)
        self.tgt_len = torch.Tensor([len(l) for l in self.spectra]).to(torch.int32)
        self.max_src_seq_len = int(max(self.src_len))
        self.max_tgt_seq_len = int(max(self.tgt_len))

        self.spectra = torch.Tensor(self.spectra)

    def __len__(self):
        return len(self.norm_normal00)

    def __getitem__(self, index):
        paras = self.norm_normal00[index]
        result = self.spectra[index]
        return paras, result

    def print(self):
        print('class GoldNanorodDelta with: ')
        print('data path = {}'.format(self.data_path))
        print('rods = {}'.format(self.rods))
        print('device = {}'.format(self.device))
        print('use TL_TR = {}'.format(self.use_TL_TR))
        print('make spectrum integer = {}'.format(self.make_spectrum_int))
        print('transform = {}'.format(self.transform))
        print('spectrum sample rate = {}'.format(self.sample_rate))
        print('source size = [{}, {}]'.format(len(self.norm_normal00), self.max_src_seq_len))
        print('target size = [{}, {}]'.format(len(self.spectra), self.max_tgt_seq_len))

    def print_item(self, index):
        print('source = {}'.format(self.norm_normal00[index]))
        print('target = {}'.format(self.spectra[index]))
