import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

from data_helper import get_data_complete

sys.path.append(os.path.dirname(os.path.abspath('.'))+'/dl_helper')
from utils import n_t


# '20181023', '20181026', '20181031', '20181115', '20181123', '20190129',
# '20190130', '20190201', '20190208', '20190220', '20190227']

class FLImDataset(Dataset):

    FILENAME = 'all_patient.pickle'

    def __init__(
            self,
            datatype='it',
            transforms=None,
            path='../data/processed/',
            cleaned=True,
            band=True,
            clean_prefix='MDCEBL',
            shape=128,
            seed=42,
            n_img=-1
    ):
        self.seed = seed

        self.transforms = transforms

        self.read_data(path, cleaned, band, clean_prefix)
        self.format_data(datatype)
        self.reshape_data(shape)
        self.cut_data(n_img)
        self.to_tensor()

        self.idx = np.arange(len(self.data))
        self.in_channels = self.data.shape[1]

        self.state = 'train'

    def validate(self):
        self.state = 'val'

    def train(self):
        self.state = 'train'

    def test(self):
        self.state = 'test'

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
        self.band = band.astype(int)

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

    def reshape_data(self, shape):
        if len(self.data.shape) == 2:
            self.data = self.data.reshape(*self.data.shape, 1)
        self.data = self.data.reshape(self.data.shape[0], shape,
                                      shape, self.data.shape[2])

    def cut_data(self, n_img):
        if n_img != -1:
            self.data = self.data[:n_img]
            self.label = self.label[:n_img]
            self.band = self.band[:n_img]
            self.patient = self.patient[:n_img]

    def to_tensor(self):
        self.data = torch.from_numpy(self.data).float()
        self.label = torch.from_numpy(self.label)
        # self.patient = torch.from_numpy(self.patient)
        self.band = torch.from_numpy(self.band)
        self.data = n_t(self.data, b=True)

    def __len__(self):
        x, _, _, _ = self.get_current_data()
        return len(x)

    def __getitem__(self, idx):
        x, y, p, b = self.get_current_data()

        x = x[idx]
        y = y[idx]
        p = p[idx]
        b = b[idx]

        if self.transforms is not None and self.state == 'train':
            x, = self.transforms((x,))

        return x, y, p, b

    def get_current_data(self, idx=None):
        if idx is None:
            idx = self.idx
        return self.data[idx], self.label[idx], \
            self.patient[idx], self.band[idx]
