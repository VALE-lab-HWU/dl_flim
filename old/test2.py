import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2gray

import torch
from torch import nn
import kornia as K
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import timm

import test22

def read_histo_from_path(path):
    print(f"read from {path}")
    names = []
    labels = []
    for i in os.listdir(path):
        tmp = os.listdir(path+'/'+i)
        names.extend(tmp)
        labels.extend([i] * len(tmp))
    data = []
    new_labels = []
    for i, name in enumerate(names):
        print(f"{i+1}/{len(names)}      ", end="\r")
        read = np.array(Image.open(f"{path}/{labels[i]}/{name}"))
        if read.shape == (50, 50, 3):
            data.append(rgb2gray(read))
            new_labels.append(int(labels[i][0]))
    data = torch.tensor(data)
    labels = torch.tensor(new_labels)
    print('')
    return data, labels

def read_img(path='/train'):
    path = "../data/Breast Histopathology Images SMALL"+path
    x, y = read_histo_from_path(path)
    return reshape(x), y


def read_histo_small():
    path = "../data/Breast Histopathology Images SMALL"
    train_path = path+'/train'
    val_path = path+'/valid'
    test_path = path+'/test'
    x_train, y_train = read_histo_from_path(train_path)
    x_val, y_val = read_histo_from_path(val_path)
    x_test, y_test = read_histo_from_path(test_path)
    return reshape(x_train), y_train, reshape(x_val), y_val,reshape(x_test), y_test


def reshape(x):
    return x.reshape(x.shape[0], 1, *x.shape[1:])


x_train, y_train = read_img('/train') 
x_val, y_val = read_img('/valid')
x_test, y_test = read_img('/test')

# Taken from here https://stackoverflow.com/a/58748125/1983544
import os
num_workers = os.cpu_count() 
if 'sched_getaffinity' in dir(os):
    num_workers = len(os.sched_getaffinity(0))
print (num_workers)

batch_size = 32
train_dataloader = DataLoader(list(zip(x_train, y_train)),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers = num_workers)
valid_dataloader = DataLoader(list(zip(x_val, y_val)),
                              batch_size=batch_size,
                              shuffle=False,
                              drop_last=False,
                              num_workers = num_workers)

model = timm.create_model('resnet50', pretrained=True)

model.default_cfg

model = timm.create_model('resnet50', pretrained=True, num_classes=10)
print(model.fc)

loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
lambda_l2 = 1e-4
optimizer = torch.optim.Adam(model.fc.parameters(),
                             lr=learning_rate,
                             eps=1e-2,
                             weight_decay=lambda_l2)  # built-in L2

epochs = 1

test22.train_epochs(train_dataloader, valid_dataloader, model, loss_fn, optimizer, epochs=epochs)
