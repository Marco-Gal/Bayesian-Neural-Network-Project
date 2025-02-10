# @Author: amishkin
# @Date:   18-08-01
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   amishkin
# @Last modified time: 18-08-23

import os
import warnings
import torch
import numpy as np
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

import sklearn.model_selection as modsel


DEFAULT_DATA_FOLDER = "~/data"


################################################
## Construct class for dealing with data sets ##
################################################

class Dataset():
    def __init__(self, data_set, data_folder=DEFAULT_DATA_FOLDER):
        super(type(self), self).__init__()

        if data_set == "mnist":
            self.train_set = dset.MNIST(root = data_folder,
                                        train = True,
                                        transform = transforms.ToTensor(),
                                        download = True)

            self.test_set = dset.MNIST(root = data_folder,
                                       train = False,
                                       transform = transforms.ToTensor())

            self.task = "classification"
            self.num_features = 28 * 28
            self.num_classes = 10

        if data_set == "mnist_val":
            full_train_set = dset.MNIST(root = data_folder,
                                        train = True,
                                        transform = transforms.ToTensor(),
                                        download = True)

            self.train_set = torch.utils.data.Subset(full_train_set, np.arange(start=0, stop=50000))

            self.test_set = torch.utils.data.Subset(full_train_set, np.arange(start=50000, stop=60000))

            self.task = "classification"
            self.num_features = 28 * 28
            self.num_classes = 10

        elif data_set == "smnist":
            self.test_set = dset.MNIST(root = data_folder,
                                       train = True,
                                       transform = transforms.ToTensor(),
                                       download = True)

            self.train_set = dset.MNIST(root = data_folder,
                                        train = False,
                                        transform = transforms.ToTensor())

            self.task = "classification"
            self.num_features = 28 * 28
            self.num_classes = 10

       


    def get_train_size(self):
        return len(self.train_set)

    def get_test_size(self):
        return len(self.test_set)

    def get_train_loader(self, batch_size, shuffle=True):
        train_loader = DataLoader(dataset = self.train_set,
                                  batch_size = batch_size,
                                  shuffle = shuffle)
        return train_loader

    def get_test_loader(self, batch_size, shuffle=False):
        test_loader = DataLoader(dataset = self.test_set,
                                 batch_size = batch_size,
                                 shuffle = shuffle)
        return test_loader

    def load_full_train_set(self, use_cuda=torch.cuda.is_available()):

        full_train_loader = DataLoader(dataset = self.train_set,
                                       batch_size = len(self.train_set),
                                       shuffle = False)

        x_train, y_train = next(iter(full_train_loader))

        if use_cuda:
            x_train, y_train = x_train.cuda(), y_train.cuda()

        return x_train, y_train

    def load_full_test_set(self, use_cuda=torch.cuda.is_available()):

        full_test_loader = DataLoader(dataset = self.test_set,
                                      batch_size = len(self.test_set),
                                      shuffle = False)

        x_test, y_test = next(iter(full_test_loader))

        if use_cuda:
            x_test, y_test = x_test.cuda(), y_test.cuda()

        return x_test, y_test



#######################################################################
## Construct class for dealing with data sets using cross-validation ##
#######################################################################

class DatasetCV():
    def __init__(self, data_set, n_splits=3, seed=None, data_folder=DEFAULT_DATA_FOLDER):
        super(type(self), self).__init__()

        self.n_splits = n_splits
        self.seed = seed
        self.current_split = 0

        if data_set == "mnist":
            self.data = dset.MNIST(root = data_folder,
                                   train = True,
                                   transform = transforms.ToTensor(),
                                   download = True)

            self.task = "classification"
            self.num_features = 28 * 28
            self.num_classes = 10

        elif data_set == "smnist":
            self.data = dset.MNIST(root = data_folder,
                                   train = False,
                                   transform = transforms.ToTensor(),
                                   download = True)

            self.task = "classification"
            self.num_features = 28 * 28
            self.num_classes = 10


        else:
            RuntimeError("Unknown data set")

        # Store CV splits
        cv = modsel.KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = cv.split(range(len(self.data)))

        self.split_idx_val = []
        for (_, idx_val) in splits:
            self.split_idx_val.append(idx_val)

    def get_full_data_size(self):
        return len(self.data)

    def get_current_split(self):
        return self.current_split

    def set_current_split(self, split):
        if split >= 0 and split <= self.n_splits-1:
            self.current_split = split
        else:
            RuntimeError("Split higher than number of splits")

    def _get_current_val_idx(self):
        return self.split_idx_val[self.current_split]

    def _get_current_train_idx(self):
        return np.setdiff1d(range(len(self.data)), self.split_idx_val[self.current_split])

    def get_current_val_size(self):
        return len(self._get_current_val_idx())

    def get_current_train_size(self):
        return len(self.data) - len(self._get_current_val_idx())

    def get_current_train_loader(self, batch_size, shuffle=True):
        train_set = torch.utils.data.Subset(self.data, self._get_current_train_idx())
        train_loader = DataLoader(dataset = train_set,
                                  batch_size = batch_size,
                                  shuffle = shuffle)
        return train_loader

    def get_current_val_loader(self, batch_size, shuffle=True):
        val_set = torch.utils.data.Subset(self.data, self._get_current_val_idx())
        val_loader = DataLoader(dataset = val_set,
                                batch_size = batch_size,
                                shuffle = shuffle)
        return val_loader

    def load_current_train_set(self, use_cuda=torch.cuda.is_available()):
        x_train, y_train = self.data.__getitem__(self._get_current_train_idx())

        if use_cuda:
            x_train, y_train = x_train.cuda(), y_train.cuda()

        return x_train, y_train

    def load_current_val_set(self, use_cuda=torch.cuda.is_available()):
        x_test, y_test = self.data.__getitem__(self._get_current_val_idx())

        if use_cuda:
            x_test, y_test = x_test.cuda(), y_test.cuda()

        return x_test, y_test