import os
import sys

sys.path.append(os.path.dirname(os.path.abspath('.'))+'/lime')

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import dl_helper as dlh
import torch
import numpy as np
import ml_helper as mlh
from PIL import Image
from sklearn.metrics import accuracy_score
from skimage.color import rgb2gray


def get_hyperparameter(model):
    loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate = 1e-3
    lambda_l2 = 1e-5
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=lambda_l2)  # built-in L2
    return loss_fn, optimizer


def one():
    """
    data from the Labeled Faces in the Wild (lfw) dataset
    """
    fw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    data = fw_people.images.reshape(len(fw_people.data), 1, 50, 37)
    label = fw_people.target
    x_train, x_test, y_train, y_test = train_test_split(
        data, label)
    layer_1 = dlh.create_default_2d_layer()
    pred = dlh.train_and_test(x_train, y_train, x_test, layer_1,
                              len(fw_people.target_names),
                              get_hyperparameter, dlh.DEVICE, epochs=20)
    print(accuracy_score(y_test, pred))


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
    data = np.array(data)
    labels = np.array(new_labels)
    return data, labels


def read_histo_small():
    path = "../data/Breast Histopathology Images SMALL"
    train_path = path+'/train'
    test_path = path+'/test'
    x_train, y_train = read_histo_from_path(train_path)
    x_test, y_test = read_histo_from_path(test_path)
    return x_train, y_train, x_test, y_test


def reshape(x):
    return x.reshape(x.shape[0], 1, *x.shape[1:])


def two():
    x_train, y_train, x_test, y_test = read_histo_small()
    n_class = len(np.unique(y_train))
    layer_1 = dlh.create_default_2d_layer()
    x_train = reshape(x_train)
    x_test = reshape(x_test)
    pred = dlh.train_and_test(x_train, y_train, x_test, layer_1,
                              n_class,
                              get_hyperparameter, dlh.DEVICE, epochs=20)
    print(accuracy_score(y_test, pred))
