import pandas as pd
import numpy as np
import os


PATH = '../data/processed'
FILENAME = 'cleaned_all_patient.pickle'

PATH_CLEANED = PATH + '/cleaned'
FILE_PREFIX = 'band_cleaned_'
FILE_SUFFIX = '_all_patient.pickle'
FILEMAIN = 'MDCEBL'

FILENAME = FILE_PREFIX + FILEMAIN + FILE_SUFFIX


# read the data in the file located at the filepath
# dic is an option because some pickle file, when loaded, are inside dic
# return a DataFrame
def read_data_pickle(filepath=PATH+'/'+FILENAME, dic=False):
    data = pd.read_pickle(filepath)
    if dic:
        data = [data[i] for i in data][0]
    return data


# read the data in the file located at the filepath
# dic is an option because some pickle file, when loaded, are inside dic
# return a DataFrame
def read_data_csv(filepath=PATH+'/'+FILENAME):
    return pd.read_csv(filepath)


# read all the data in the folder path
# if the subfolder option is True then the file are assumed to be in subfolder
# of the same name (starting with the sate, so 20XX)
# return an array of DataFrame
def read_all_data(path=PATH, subfolder=True):
    listdir = os.listdir(path)
    listdir.sort()
    res = []
    for folder in listdir:
        if subfolder:
            if folder[0] == '2':
                print(folder)
                for files in os.listdir(path+'/'+folder):
                    if files[-7:] == '.pickle':
                        data = read_data_pickle(path+'/'+folder+'/'+files,
                                                True)
                        data['patient'] = files[:8]
                        res.append(data)
        else:
            if folder[-7:] == '.pickle' and folder[0] == '2':
                data = read_data_pickle(path+'/'+folder, False)
                data['patient'] = folder[:8]
                res.append(data)
    return res


# read all the pickle files and concat them in one DataFrame
def concat_data(path=PATH, subfolder=True):
    data = read_all_data(path, subfolder)
    res = pd.concat(data, ignore_index=True)
    return res


# concat all the different files in the PATH folder, and concat them in
# the file named filename
def write_data(path=PATH, filename=FILENAME, subfolder=True):
    data = concat_data(path, subfolder)
    data.to_pickle(path+'/'+filename)


# extract the lifetime from a dataframe
def lifetime_of_data(data):
    return data[data.columns[16401:32785]]


# extract the intensity from a dataframe
def intensity_of_data(data):
    return data[data.columns[17:16401]]


def extract_informative_feature(data, feature):
    return data[feature].to_numpy()


# extract the patient from a dataframe
def extract_patient(data):
    return extract_informative_feature(data, 'patient')


# extract the band from a dataframe
def extract_band(data):
    return extract_informative_feature(data, 'band')


# extract the label from a dataframe
def extract_label(data):
    return extract_informative_feature(
        data, 'tissue_classification').astype(int)


# extract a feature from a dataframe
# alias to change easily, only once
def extract_feature(data):
    return intensity_of_data(data).to_numpy()


# extract a lifetine and intensity from a dataframe
def extract_features(data):
    return (intensity_of_data(data).to_numpy(),
            lifetime_of_data(data).to_numpy())


# extract the label, lifetime and intensity of a files
def get_datas(path=PATH, filename=FILENAME):
    data = read_data_pickle(path+'/'+filename, False)
    intensity, lifetime = extract_features(data)
    label = extract_label(data)
    return intensity, lifetime,  label


# extract the intensity from a file
def get_intensity(path=PATH, filename=FILENAME):
    data = read_data_pickle(path+'/'+filename, False)
    return intensity_of_data(data).to_numpy()


# etract the lifetime from a file
def get_lifetime(path=PATH, filename=FILENAME):
    data = read_data_pickle(path+'/'+filename, False)
    return lifetime_of_data(data).to_numpy()


# extract data from all files
# return the array of array of feature and label extracted
def get_data_per_files(path):
    data = read_all_data(path=PATH)
    label = np.array([extract_label(d) for d in data])
    data = np.array([extract_feature(d) for d in data])
    return data, label


# read a file and return the array of label and feature extracted
def get_data_complete(path=PATH, filename=FILENAME,
                      all_feature=False, feature='lf'):
    if filename not in os.listdir(path):
        raise Exception('File not found')
        # write_data(path, filename)
    print('Read data')
    data = read_data_pickle(path+'/'+filename, False)
    label = extract_label(data)
    patient = extract_patient(data)
    band = extract_band(data)
    if all_feature:
        data = extract_features(data)
    else:
        if feature == 'lf':
            data = lifetime_of_data(data).to_numpy()
        elif feature == 'it':
            data = intensity_of_data(data).to_numpy()
        else:
            data = extract_feature(data)
    return data, label, patient, band


def get_patient_dict(patient):
    p = np.unique(patient)
    return {i: p[i] for i in range(len(p))}
