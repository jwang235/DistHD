import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim


from time import time

import torch
import sklearn.datasets
import sklearn.preprocessing
import sklearn.model_selection
import numpy as np

import pandas as pd
from unlzw import unlzw
import unlzw3
from pathlib import Path
import csv


def mnist():
    x, y = sklearn.datasets.fetch_openml('mnist_784', return_X_y=True)
    x, y = x.astype(np.float), y.astype(np.long)
    x, x_test, y, y_test = sklearn.model_selection.train_test_split(x, y)
    scaler = sklearn.preprocessing.Normalizer().fit(x)
    x_train = scaler.transform(x)
    x_test = scaler.transform(x_test)
    
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.tensor(y.values.ravel())
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.tensor(y_test.values.ravel())

    n_classes = len(y.unique())
    n_features = x.shape[1]
    X, Y = (x_train, x_test), (y_train, y_test)
    return X, Y, n_classes, n_features

def isolet():
    train = pd.read_csv('data/ISOLET/isolet_train.csv', sep=',', header=None)
    test = pd.read_csv('data/ISOLET/isolet_test.csv', sep=',', header=None)
    train, test = torch.tensor(train.values), torch.tensor(test.values)

    x_train = train[:, 0: (train.shape[1]-1)].float()
    y_train = (train[: ,-1]-1).long()
   #  x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x_train, y_train)
    x_test = test[:, 0: (train.shape[1]-1)].float()
    y_test = (test[: ,-1]-1).long()

    n_classes = len(torch.unique(torch.cat((y_train, y_test))))
    n_features = x_train.shape[1]
    X, Y = (x_train, x_test), (y_train, y_test)
    return X, Y, n_classes, n_features

def ucihar():
    x_train = pd.read_csv('data/UCIHAR/X_train.txt', sep='\s+', header=None)
    x_test = pd.read_csv('data/UCIHAR/X_test.txt', sep='\s+', header=None)
    y_train = pd.read_csv('data/UCIHAR/y_train.txt', sep='\s+', header=None)
    y_test = pd.read_csv('data/UCIHAR/y_test.txt', sep='\s+', header=None)
    x_train, x_test = torch.tensor(x_train.values).float(), torch.tensor(x_test.values).float()
    y_train, y_test = (torch.tensor(y_train.values)-1).long().squeeze(), (torch.tensor(y_test.values)-1).long().squeeze()
    X, Y = (x_train, x_test), (y_train, y_test)
    n_classes = len(torch.unique(torch.cat((y_train, y_test))))
    n_features = x_train.shape[1]
    # print(torch.unique(torch.cat((y_train, y_test))))
    return X, Y, n_classes, n_features

def pamap2():
    data = torch.tensor(pd.read_csv('data/PAMAP2/pamap2.csv').values)
    x, y = data[ : , 0 : data.shape[1]-1].float(), data[ : , -1].long()
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x , y)
    n_classes = len(torch.unique(torch.cat((y_train, y_test))))
    n_features = x_train.shape[1]
    X, Y = (x_train, x_test), (y_train, y_test)
    return X, Y, n_classes, n_features

