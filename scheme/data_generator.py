"""
Defines the Dataset, DevDataLoader and EvalDataLoader classes.

"""

import h5py
import numpy as np
import torch
from scheme.utils import get_train_valid_indices, get_standard_params


class Dataset(object):

    def __init__(self, hdf5_path, dev=True, split_indices_csv_path=None, label_type=None):
        data = h5py.File(hdf5_path, 'r')
        self.x = data['feature'][:]
        self.n_samples = np.shape(self.x)[0]
        self.identifier = np.array([s.decode() for s in data['identifier']])
        self.alarm_ids = np.array([s.decode() for s in data['alarm_id']])
        self.background_id = np.array([s.decode() for s in data['background_id']])
        self.noise_level = data['noise_level'][:]
        self.snr = data['snr'][:]
        self.level_correction = data['level_correction'][:]
        self.alarm_onset = data['alarm_onset'][:]
        self.annotators = data['annotators'][:]

        if dev is True:
            self.y = data['label'][:]
            if split_indices_csv_path is not None:
                self.train_indices, self.valid_indices = get_train_valid_indices(split_indices_csv_path)
            else:
                self.train_indices = np.arange(len(self))
        elif dev is False:
            if label_type == 'mv':
                self.y = (data['label_mv'][:] > 0.5).astype(int)
            elif label_type == 'apf':
                self.y = data['label_apf'][:]
            else:
                raise ValueError("Variable label_type must be 'mv' or 'apf'.")
        else:
            raise ValueError("Variable 'dev' must be True or False.")

        data.close()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class DevDataLoader(object):

    def __init__(self, dataset, batch_size, seed=199):
        self.dataset = dataset
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)
        self.mean, self.std = get_standard_params(self.dataset.x[self.dataset.train_indices])

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        x = self.transform(x.numpy())
        return x, y

    def transform(self, x):
        if x.ndim == 3:
            return (x - self.mean) / self.std
        elif x.ndim == 4:
            x_ = x
            for i in range(x.shape[1]):
                x_[:, i] = (x_[:, i] - self.mean[i]) / self.std[i]
            return x_

    def generate(self, dev_subset='training', annotator_indices=None, bin_threshold=0.5):
        if dev_subset == 'training':
            indices = self.dataset.train_indices
            audios_num = len(indices)
            self.random_state.shuffle(indices)
        elif dev_subset == 'validation':
            indices = self.dataset.valid_indices
            audios_num = len(indices)
        else:
            raise ValueError("Variable 'dev_subset' must be 'training' or 'validation'.")

        pointer = 0
        while pointer < audios_num:
            batch_id = indices[pointer:pointer+self.batch_size]
            pointer += self.batch_size

            batch_x = self.dataset.x[batch_id][:]
            if annotator_indices is not None:
                annotators = annotator_indices[batch_id]
                batch_y = self.dataset.y[batch_id[:, np.newaxis], annotators]
            else:
                batch_y = self.dataset.y[batch_id][:]
            batch_y = np.mean(batch_y, axis=1)

            batch_y = (batch_y > bin_threshold).astype(int)
            batch_y = torch.LongTensor(batch_y)

            batch_x = self.transform(batch_x)
            batch_x = torch.FloatTensor(batch_x).view(batch_x.shape[0], -1, batch_x.shape[-2], batch_x.shape[-1])

            yield batch_x, batch_y


class EvalDataLoader(object):

    def __init__(self, dataset, batch_size, training_mean=None, training_std=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.training_mean = training_mean
        self.training_std = training_std

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        x = self.transform(x.numpy())
        return x, y

    def transform(self, x):
        if x.ndim == 3:
            return (x - self.training_mean) / self.training_std
        elif x.ndim == 4:
            x_ = x
            for i in range(x.shape[1]):
                x_[:, i] = (x_[:, i] - self.training_mean[i]) / self.training_std[i]
            return x_

    def generate(self, annotator_indices=None, bin_threshold=0.5):
        audios_num = len(self.dataset)
        indices = np.arange(audios_num)

        pointer = 0
        while pointer < audios_num:
            batch_id = indices[pointer:pointer + self.batch_size]
            pointer += self.batch_size

            batch_x = self.dataset.x[batch_id][:]
            if annotator_indices is not None:
                annotators = annotator_indices[batch_id]
                batch_y = self.dataset.y[batch_id[:, np.newaxis], annotators]
            else:
                batch_y = self.dataset.y[batch_id]
            batch_y = np.mean(batch_y, axis=1)

            batch_y = (batch_y > bin_threshold).astype(int)
            batch_y = torch.LongTensor(batch_y)

            batch_x = self.transform(batch_x)
            batch_x = torch.FloatTensor(batch_x).view(batch_x.shape[0], -1, batch_x.shape[-2], batch_x.shape[-1])

            yield batch_x, batch_y
