import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_helper import get_data_complete


class FlimDataset(Dataset):

    FILENAME = 'all_patient.pickle'

    def __init__(
            self,
            datatype='it',
            path='../data/processed/',
            cleaned=True,
            band=True,
            clean_prefix='MDCEBL',
            seed=42
    ):
        self.seed = seed

        self.read_data(path, cleaned, band, clean_prefix)
        self.format_data(datatype)

    def read_data(self, path, cleaned, band, prefix):
        name = prefix + '_' + self.FILENAME
        if cleaned:
            path += 'cleaned/'
            name = 'cleaned_' + name
        if band:
            name = 'band_' + name
        data, label, patient, band = get_data_complete(
            path=path, filename=name, all_feature=True)
        self.data = data
        self.label = label
        self.patient = patient
        self.band = band

    def get_set_lf(self):
        self.data = self.data[1]

    def get_set_it(self):
        self.data = self.data[0]

    def get_set_it_lf(self):
        self.data = np.dstack(self.data)

    def get_set_band(self, band):
        self.data = self.data[self.band == band]
        self.label = self.label[self.band == band]
        self.patient = self.patient[self.band == band]
        self.band = self.band[self.band == band]

    def get_set_b1(self):
        self.get_set_band(band=1)

    def get_set_b2(self):
        self.get_set_band(band=2)

    def get_set_b1_b2(self):
        idx = self.band == 1
        self.label = self.label[idx]
        self.patient = self.patient[idx]
        self.band[idx] = 3
        self.band = self.band[idx]
        self.data = np.dstack((self.data[idx], self.data[~idx]))

    def get_set_lf_b1(self):
        self.get_set_lf()
        self.get_set_b1()

    def get_set_lf_b2(self):
        self.get_set_lf()
        self.get_set_b2()

    def get_set_it_b1(self):
        self.get_set_it()
        self.get_set_b1()

    def get_set_it_b2(self):
        self.get_set_it()
        self.get_set_b2()

    def get_set_it_lf_b1(self):
        self.get_set_it_lf()
        self.get_set_b1()

    def get_set_it_lf_b2(self):
        self.get_set_it_lf()
        self.get_set_b2()

    def get_set_it_b1_b2(self):
        self.get_set_it()
        self.get_set_b1_b2()

    def get_set_lf_b1_b2(self):
        self.get_set_lf()
        self.get_set_b1_b2()

    def get_set_it_lf_b1_b2(self):
        self.get_set_it_lf()
        self.get_set_b1_b2()

    def format_data(self, datatype):
        fns = {
            'lf': self.get_set_lf,
            'it': self.get_set_it,
            'lf_b1': self.get_set_lf_b1,
            'lf_b2': self.get_set_lf_b2,
            'it_b1': self.get_set_it_b1,
            'it_b2': self.get_set_it_b2,
            'lf_b1_b2': self.get_set_lf_b1_b2,
            'it_b1_b2': self.get_set_it_b1_b2,
            'it_lf_b1': self.get_set_it_lf_b1,
            'it_lf_b2': self.get_set_it_lf_b2,
            'it_lf_b1_b2': self.get_set_it_lf_b1_b2
        }
        if datatype not in fns:
            raise Exception('wrong data type')
        fns[datatype]()
