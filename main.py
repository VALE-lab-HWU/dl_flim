import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision.models import get_model as get_TF_model
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split, KFold
from functools import partial

from arg import parse_args
from dataset import FlimDataset
from utils import log, store_results
from ml_helper import compare_class
from transform import get_transforms
import dl_helper


def test_model(model, ts_dl, title):
    y_pred, y_true = dl_helper.test(ts_dl, model, device=device)
    y_pred = torch.argmax(y_pred, dim=1)
    compare_class(y_pred, y_true)


def main(args):
    device = torch.device("cpu" if not torch.cuda.is_available()
                          else args.device)
    log(f'Device: {device}', args.log, 1)
    log('Create dataset, dataloader, model', args.log, 1)
    tr_dl, v_dl, ts_dl = get_data_loader(args)
    model = get_model(args, tr_dl.dataset.in_channels)
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(args, model)
    model, l_tt, l_vt = dl_helper.train_epochs(
        tr_dl, v_dl, model, loss_fn,
        optimizer, log=args.log,
        epochs=args.md_epochs, device=device)
    best_model = get_model(
        args, tr_dl.dataset.in_channels)
    best_model = dl_helper.load_model(args.title, best_model,
                                      device=device)
    test_model(model, ts_dl, 'Last model')
    test_model(best_model, ts_dl, 'Best model')    
    store_results(l_tt=l_tt, l_vt=l_vt, title=args.title+'_loss')


def get_optimizer(args, model):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.md_learning_rate)
    return optimizer


def get_model(args, in_channels):
    md = get_TF_model('ResNet50')
    md.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                               padding=(3, 3), bias=False)
    md.fc = torch.nn.Linear(in_features=2048, out_features=2, bias=True)
    return md


def get_idx_split_or_patient(arg, idx, patient, shuffle):
    if type(arg) is str:
        if arg  in patient:
            tmp = patient == arg
            idx_one = idx[(~tmp).nonzero()[0]]
            idx_two = idx[tmp.nonzero()[0]]
        else:
            raise Exception('Patient is not in list of patient')
    else:
        idx_one, idx_two = train_test_split(idx, shuffle=shuffle,
                                               test_size=arg) 
    return idx_one, idx_two


def get_idx_test(args, n, patient):
    idx = np.arange(n)
    train_idx, test_idx = get_idx_split_or_patient(args.dl_val_subset, idx,
                                                   patient, args.dl_split_shuffle)
    train_idx, val_idx = get_idx_split_or_patient(args.dl_val_subset,
                                                  train_idx, patient[train_idx], args.dl_split_shuffle)
    return train_idx, val_idx, test_idx


def get_data_loader_test(dataset, args):
    tr_idx, v_idx, ts_idx = get_idx_test(args, len(dataset), patient)
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train_dl = DataLoader(
        dataset,
        batch_size=args.dl_batch_size,
        shuffle=args.dl_shuffle,
        sampler=train_sampler)
    val_dl = DataLoader(
        dataset,
        batch_size=args.dl_batch_size,
        shuffle=args.dl_shuffle,
        sampler=val_sampler)
    test_dl = DataLoader(
        dataset,
        batch_size=args.dl_batch_size,
        shuffle=args.dl_shuffle,
        sampler=test_sampler)
    return train_dl, val_dl, test_dl


def get_k_fold_split(dataset, k, shuffle):
    kf = KFold(n_splits=k, shuffle=shuffle)
    return {i: v for i, v in enumerate(kf.split(np.arange(len(dataset))))}


def get_split_per_patients(dataset):
    return {i: (dataset.patient==i).nonzero()[0] for i in np.unique(dataset.patient)}


def get_data_loaders_cross(args, dataset):
    if args.k_cross:
        return get_k_fold_split(dataset, args.cross_nb, args.dl_split_shuffle)
    elif args.p_cross:
        return get_split_per_patients(dataset)
    else:
        raise Exception("Something unexpected happen. Cross-validation argument are wrong.")


def get_data_loader(args):
    dataset = get_dataset(args)
    if args.cross:
        return get_data_loaders_cross(dataset, args)
    else:
        return get_data_loader_test(dataset, args)


def get_dataset(args):
    transforms = partial(
        get_transforms,
        angle=args.tf_angle,
        flip_prob=args.tf_flip,
    )
    dataset = FlimDataset(
        n_img=args.ds_n_img,
        seed=args.seed,
        transforms=transforms,
    )
    return dataset


if __name__ == '__main__':
    args = parse_args('FLIm dl')
    args.cross = args.k_cross or args.p_cross
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)
