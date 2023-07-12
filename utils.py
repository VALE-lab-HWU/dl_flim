import torch
import pandas as pd


def log(msg, log_lv, log=0):
    if log_lv >= log:
        print(msg)


def t_n(tens, b=False):
    arange = torch.arange(tens.ndim)
    if b:
        return tens.permute(0, *arange[-2:], *arange[1:-2])
    else:
        return tens.permute(*arange[-2:], *arange[:-2])


def n_t(tens, b=False):
    arange = torch.arange(tens.ndim)
    if b:
        return tens.permute(0, *arange[3:], *arange[1:3])
    else:
        return tens.permute(*arange[2:], *arange[:2])


def store_results(title='flim', name='result', **res):
    pd.to_pickle(res, f'./results/{title}/{name}.pkl')
