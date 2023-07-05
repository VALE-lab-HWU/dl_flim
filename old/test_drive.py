import h5py
import os
import numpy as np
import matplotlib.pyplot as plt



PATH = '/media/jackred/Seagate Expansion Drive/FS-FLIM/inverted/'

list_dir = os.listdir(PATH)
list_dir = [i for i in list_dir if '.' not in i]

folder = list_dir[0]
list_folder = os.listdir(PATH+'/'+folder+'/HistMode_no_pixel_binning')

fol1 = list_folder[0]
list_files = os.listdir(PATH+'/'+folder+'/HistMode_no_pixel_binning/'+fol1)

file1 = list_files[0]

m = h5py.File(PATH+'/'+folder+'/HistMode_no_pixel_binning/'+fol1+'/'+file1, 'r')
d = np.array(m.get('bins_array_3'))
