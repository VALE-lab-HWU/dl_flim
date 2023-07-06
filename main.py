import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision.models import get_model as get_TF_model
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split

from arg import parse_args
from dataset import FlimDataset
from utils import log
import dl_helper


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


def get_optimizer(args, model):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.md_learning_rate)
    return optimizer


def get_model(args, in_channels):
    md = get_TF_model('ResNet50')
    md.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                               padding=(3, 3), bias=False)
    return md


def get_data_loader(args):
    dataset = get_dataset(args)
    idx = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(idx, shuffle=args.dl_split_shuffle,
                                           test_size=args.dl_test_subset)
    train_idx, val_idx = train_test_split(train_idx,
                                          shuffle=args.dl_split_shuffle,
                                          test_size=args.dl_val_subset)
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.dl_batch_size,
        shuffle=args.dl_shuffle,
        sampler=train_sampler)
    val_dataloader = DataLoader(
        dataset,
        batch_size=args.dl_batch_size,
        shuffle=args.dl_shuffle,
        sampler=val_sampler)
    test_dataloader = DataLoader(
        dataset,
        batch_size=args.dl_batch_size,
        shuffle=args.dl_shuffle,
        sampler=test_sampler)
    return train_dataloader, val_dataloader, test_dataloader


def get_dataset(args):
    dataset = FlimDataset(
        seed=args.seed,
    )
    return dataset


if __name__ == '__main__':
    args = parse_args('FLIm dl')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)
