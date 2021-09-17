import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath('.'))+'/lime')

import lime_helper as lh
import process_helper as ph
import data_helper as dh
import model_helper as mh
import explain_helper as eh
import ml_helper as mlh
import ml_helper as mlh
import data_helper as dh
import dl_helper as dlh

PATH = dh.PATH+'/cleaned'
FILE_PREFIX = 'processed_cleaned_'
FILE_SUFFIX = '_all_patient.pickle'
FILEMAIN = 'MDCEBL'

FILENAME = FILE_PREFIX + FILEMAIN + FILE_SUFFIX


def get_hyperparameter(model):
    loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate = 1e-3
    lambda_l2 = 1e-5
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=lambda_l2)  # built-in L2
    return loss_fn, optimizer



def main(path=PATH, filename=FILENAME, device=dlh.DEVICE):
    data, label, patient, band = dh.get_data_complete(
        path, filename, False, feature='it')
    data = data.reshape(-1, 1, 128, 128)
    id1 = band == 1
    id2 = band == 2
    d1, l1, p1 = data[id1], label[id1], patient[id1]
    d2, l2, p2 = data[id2], label[id2], patient[id2]
    x_train, x_test, y_train, y_test = train_test_split(d1, l1)
    pred = dlh.train_and_test(x_train, y_train, x_test, get_hyperparameter, device, epochs=3)


if __name__ == '__main__':
    main()

A = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
B = [[4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
