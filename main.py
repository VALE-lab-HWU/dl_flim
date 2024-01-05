import os
import sys
import numpy as np
import torch
import time

from torch.utils.data import DataLoader
from torchvision.models import get_model as get_TF_model
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split, KFold, LeaveOneGroupOut
from functools import partial

from arg import parse_args
from dataset import FLImDataset
import dl_helper

sys.path.append(os.path.dirname(os.path.abspath('.'))+'/dl_helper')
from utils import log, store_results, mkdir
from ml_helper import compare_class
from transform import get_transforms
from models.paper.resnet import get_resnet_50_flim
from models.paper.convnext import get_convnext_flim


MD_DICT = {'resnet': get_resnet_50_flim, 'convnext': get_convnext_flim}


def test_model_fn(model, ts_dl, title, name,  device):
    print(f'Testing {title} {name}')
    y_pred, y_true = dl_helper.test(ts_dl, model, device=device)
    y_pred = torch.argmax(y_pred, dim=1).contiguous().view(-1).cpu()
    y_true = y_true.contiguous().view(-1).cpu()
    compare_class(y_pred, y_true, unique_l=[1, 0])
    store_results(y_pred=y_pred, y_true=y_true, title=title, name=name)


def test_model(model, ts_dl, title, name, cross, device):
    if cross:
        for k in model:
            test_model_fn(model[k], ts_dl, f'{title}/{k}', name, device)
    else:
        test_model_fn(model, ts_dl, title, name,  device)


def init_folder(title, add_time=True):
    mkdir('./results')
    if add_time:
        title = title + '_' + str(int(time.time()))
    if title in os.listdir('./results'):
        # if not enough then wtf
        gen = np.random.default_rng()
        title += f'_{gen.integers(42000)}'
    mkdir('./results/'+title)
    return title


def main(args):
    args.title = init_folder(args.title)
    device = torch.device("cpu" if not torch.cuda.is_available()
                          else args.device)
    log(f'Device: {device}', args.log, 1)
    log('Create dataset, dataloader, model', args.log, 1)
    tr_dl, v_dl, ts_dl = get_data_loader(args)
    if args.cross:
        for i in tr_dl:
            init_folder(f'{args.title}/{i}', add_time=False)
    model = get_model(args, ts_dl.dataset.in_channels)
    model.to(device)
    optimizer = get_optimizer(args, model)
    loss_fn = torch.nn.CrossEntropyLoss()
    model, l_tt, l_vt = dl_helper.train(
        args.cross,
        tr_dl, v_dl, model, loss_fn,
        optimizer, title=args.title, log=args.log,
        epochs=args.md_epochs, device=device)
    best_model = get_best_model(tr_dl, args, device)
    test_model(model, ts_dl, args.title,  'Last_model', args.cross,  device)
    test_model(best_model, ts_dl, args.title, 'Best_model', args.cross, device)
    store_results(l_tt=l_tt, l_vt=l_vt, title=args.title, name='loss')


def get_optimizer(args, model):
    if args.cross:
        optimizer = partial(torch.optim.Adam, lr=args.md_learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.md_learning_rate)
    return optimizer


def get_best_model(tr_dl, args, device):
    if args.cross:
        res = {}
        for k in tr_dl.keys():
            model = get_model(args, tr_dl[k].dataset.in_channels)
            res[k] = dl_helper.load_model(f'{args.title}/{k}', model,
                                          device=device)
        return res
    else:
        model = get_model(args, tr_dl.dataset.in_channels)
        return dl_helper.load_model(args.title, model, device=device)


def get_model(args, in_channels):
    md = MD_DICT[args.md_model](in_channels=in_channels,
                                out_channels=args.out_channels,
                                pretrained=args.md_pretrained)
    if args.cross:
        torch.save(md.state_dict(), f'./results/{args.title}/weights.pt')
    return md


def get_idx_split_or_patient(arg, idx, patient, shuffle):
    if type(arg) is str:
        if arg in patient:
            tmp = patient == arg
            idx_one = idx[(~tmp).nonzero()[0]]
            idx_two = idx[tmp.nonzero()[0]]
        else:
            raise Exception(f'Patient {arg} is not in list of patient')
    elif arg != 0:
        idx_one, idx_two = train_test_split(idx, shuffle=shuffle,
                                            test_size=arg)
    else:
        return idx, None
    return idx_one, idx_two


def get_idx_test(args, n, patient):
    idx = np.arange(n)
    train_idx, test_idx = get_idx_split_or_patient(args.dl_test_subset, idx,
                                                   patient,
                                                   args.dl_split_shuffle)
    train_idx, val_idx = get_idx_split_or_patient(args.dl_val_subset,
                                                  train_idx,
                                                  patient[train_idx],
                                                  args.dl_split_shuffle)
    return train_idx, val_idx, test_idx


def create_sampler(idx):
    return SubsetRandomSampler(idx)


def create_dl(dataset, args, idx):
    sampler = create_sampler(idx)
    dl = DataLoader(
        dataset,
        batch_size=args.dl_batch_size,
        shuffle=args.dl_shuffle,
        sampler=sampler)
    return dl


def get_data_loader_test(dataset, args):
    tr_idx, v_idx, ts_idx = get_idx_test(args, len(dataset), dataset.patient)
    train_dl = create_dl(dataset, args, tr_idx)
    val_dl = create_dl(dataset, args, v_idx)
    test_dl = create_dl(dataset, args, ts_idx)
    return train_dl, val_dl, test_dl


def get_k_fold_split(idx, k, shuffle):
    kf = KFold(n_splits=k, shuffle=shuffle)
    tmp = [[(i, idx[v1]), (i, idx[v2])]
           for i, (v1, v2) in enumerate(kf.split(idx))]
    return {i[0][0]: i[0][1] for i in tmp}, {i[1][0]: i[1][1] for i in tmp}


def get_split_per_patients(idx, groups):
    group = np.unique(groups)
    lg = LeaveOneGroupOut()
    tmp = [[(group[i], idx[v1]), (group[i], idx[v2])]
           for i, (v1, v2) in enumerate(lg.split(idx, groups=groups))]
    return {i[0][0]: i[0][1] for i in tmp}, {i[1][0]: i[1][1] for i in tmp}


def get_data_loaders_cross_idx(dataset, args):
    idx = np.arange(len(dataset))
    train_idx, test_idx = get_idx_split_or_patient(
        args.dl_test_subset, idx, dataset.patient, args.dl_split_shuffle)
    if args.k_cross:
        train_idx, val_idx = get_k_fold_split(train_idx, args.cross_nb,
                                              args.dl_split_shuffle)
        return train_idx, val_idx, test_idx
    elif args.p_cross:
        train_idx, val_idx = get_split_per_patients(train_idx,
                                                    dataset.patient[train_idx])
        return train_idx, val_idx, test_idx
    else:
        raise Exception("Something unexpected happen. Cross-validation"
                        + "argument are wrong.")


def get_data_loaders_cross(dataset, args):
    trs_idx, vs_idx, ts_idx = get_data_loaders_cross_idx(dataset, args)
    ts_dl = create_dl(dataset, args, ts_idx)
    trs_dl = {k: create_dl(dataset, args, trs_idx[k]) for k in trs_idx}
    vs_dl = {k: create_dl(dataset, args, vs_idx[k]) for k in vs_idx}
    return trs_dl, vs_dl, ts_dl


def get_data_loader(args):
    dataset = get_dataset(args)
    if args.cross:
        return get_data_loaders_cross(dataset, args)
    else:
        return get_data_loader_test(dataset, args)


def get_dataset(args):
    transforms = get_transforms(
        angle=args.tf_angle,
        flip_prob=args.tf_flip,
    )
    dataset = FLImDataset(
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
