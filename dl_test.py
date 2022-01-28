import os
import sys

sys.path.append(os.path.dirname(os.path.abspath('.'))+'/lime')

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import dl_helper as dlh
import torch
import ml_helper as mlh

from sklearn.metrics import accuracy_score

def get_hyperparameter(model):
    loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate = 1e-3
    lambda_l2 = 1e-5
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=lambda_l2)  # built-in L2
    return loss_fn, optimizer


def one():
    fw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    data = fw_people.images.reshape(1288, 1, 50, 37)
    label = fw_people.target
    x_train, x_test, y_train, y_test = train_test_split(
        data, label)
    layer_1 = dlh.create_default_2d_layer()
    pred = dlh.train_and_test(x_train, y_train, x_test, layer_1, 7,
                              get_hyperparameter, dlh.DEVICE, epochs=20)
    print(accuracy_score(y_test, pred))
