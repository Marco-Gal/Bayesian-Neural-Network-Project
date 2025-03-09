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
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 
import pandas as pd


import sklearn.model_selection as modsel


DEFAULT_DATA_FOLDER = "~/data"


################################################
## Construct class for dealing with data sets ##
################################################

class CustomDataset(Dataset):
    def __init__(self, X, y, train=True, transform=None):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.7, random_state=42
        )
        self.X = X_train if train else X_test
        self.y = y_train if train else y_test
        self.transform = transform
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.y[idx]


class Dataset():
    def __init__(self, data_set, data_folder=DEFAULT_DATA_FOLDER):
        super(type(self), self).__init__()

        # if data_set == "mnist":
        #     self.train_set = dset.MNIST(root = data_folder,
        #                                 train = True,
        #                                 transform = transforms.ToTensor(),
        #                                 download = True)

        #     self.test_set = dset.MNIST(root = data_folder,
        #                                train = False,
        #                                transform = transforms.ToTensor())

        #     self.task = "classification"
        #     self.num_features = 28 * 28
        #     self.num_classes = 10

        if data_set == 'mnist':
            estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = fetch_ucirepo(id=544) 
  
            X_load = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.features 
            y_load = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.targets.copy() 
            categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC','CALC','MTRANS']

            X_encoded = pd.get_dummies(X_load, columns=categorical_features, drop_first=True)
            X_encoded = X_encoded.astype({col: int for col in X_encoded.select_dtypes('bool').columns})

            X_tensor = torch.tensor(X_encoded.values, dtype=torch.float32)
            y_load['NObeyesdad'] = y_load['NObeyesdad'].astype('category').cat.codes

            # Convert to tensor
            y_tensor = torch.tensor(y_load['NObeyesdad'].to_numpy(), dtype=torch.long)

            x_tr, x_te, y_tr, y_te = train_test_split(X_tensor, y_tensor, test_size=0.7, random_state=42)
            x_mean, x_std = x_tr.mean(dim=0), x_tr.std(dim=0)
            X_tensor = (x_tr - x_mean) / x_std
            
            self.train_set = CustomDataset(X_tensor, y_tensor, train=True)#, transform=transforms.ToTensor())
            self.test_set = CustomDataset(X_tensor, y_tensor, train=False)#, transform=transforms.ToTensor())



            self.task = 'classification'
            self.num_features = 23
            self.num_classes = 7

            # self.test_set = dset.MNIST(root = data_folder,
            #                            train = False,
            #                            transform = transforms.ToTensor())


        # if data_set == "mnist_val":
        #     full_train_set = dset.MNIST(root = data_folder,
        #                                 train = True,
        #                                 transform = transforms.ToTensor(),
        #                                 download = True)

        #     self.train_set = torch.utils.data.Subset(full_train_set, np.arange(start=0, stop=50000))

        #     self.test_set = torch.utils.data.Subset(full_train_set, np.arange(start=50000, stop=60000))

        #     self.task = "classification"
        #     self.num_features = 28 * 28
        #     self.num_classes = 10
        if data_set == "mnist_val":
            estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = fetch_ucirepo(id=544) 
  
            X_load = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.features 
            y_load = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.targets.copy() 
            categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC','CALC','MTRANS']

            X_encoded = pd.get_dummies(X_load, columns=categorical_features, drop_first=True)
            X_encoded = X_encoded.astype({col: int for col in X_encoded.select_dtypes('bool').columns})

            X_tensor = torch.tensor(X_encoded.values, dtype=torch.float32)
            y_load['NObeyesdad'] = y_load['NObeyesdad'].astype('category').cat.codes

            # Convert to tensor
            y_tensor = torch.tensor(y_load['NObeyesdad'].to_numpy(), dtype=torch.long)

            full_train_set = CustomDataset(X_tensor, y_tensor, train=True)#, transform=transforms.ToTensor())
            
            self.train_set = torch.utils.data.Subset(full_train_set, np.arange(start=0, stop=500))
            self.test_set = torch.utils.data.Subset(full_train_set, np.arange(start=500, stop=633))

            self.task = "classification"
            self.num_features = 23
            self.num_classes = 7

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