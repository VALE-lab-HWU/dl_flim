import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from time import time


EPOCHS = 10
NUM_CLASSES = 2
BATCH_SIZE = 32
DEVICE = "cuda"


def load_model(title, model, name='model', device=DEVICE):
    weight = torch.load(f'./results/{title}/{name}.pt', map_location=device)
    model.load_state_dict(weight)
    return model


# save best model
def save_best_model(model, loss, title='flim'):
    if len(loss) <= 2 or (len(loss) > 2 and loss[-1] < min(loss[:-1])):
        print('saving model')
        torch.save(model.state_dict(), f'./results/{title}/model.pt')


def validate(dataloader, model, loss_fn, log=0, device=DEVICE):
    model.eval()
    model.to(device)
    dataloader.dataset.validate()
    loss = 0
    with torch.no_grad():
        for _, (data, labels, _, _) in tqdm(enumerate(dataloader),
                                            total=len(dataloader)):
            labels = labels.to(device)
            pred = model(data.to(device))
            it_loss = loss_fn(pred, labels)
            loss += float(it_loss.item())
    loss = loss / len(dataloader)
    if log == 1:
        print(f"validate loss: {loss:>7f}")
    return loss


def train_step(X, y, model, loss_fn, optimizer):
    pred = model(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # torch.cuda.empty_cache()
    return loss


# do one epoch
def train_batches(dataloader, model, loss_fn, optimizer,
                  log=0, device=DEVICE):
    model.train()
    model.to(device)
    dataloader.dataset.train()
    n_batch = len(dataloader)
    for batch, (X, y, _, _) in tqdm(enumerate(dataloader), total=n_batch):
        loss = train_step(X.to(device).float(), y.to(device), model,
                          loss_fn, optimizer)
    if log == 1:
        l_print = loss.item()
        print(f"training loss: {l_print:>7f}")
    return loss


# run through epoch
def train_epochs(tr_dl, v_dl, model, loss_fn, optimizer, log=0, title='flim',
                 epochs=EPOCHS, device=DEVICE):
    loss_val = validate(v_dl, model, loss_fn, log=log,
                        device=device)
    loss_train_time = []
    loss_val_time = [loss_val]
    i = 0
    while i < epochs:
        t = time()
        print(f"Epoch {i+1}\n----------------------------")
        loss = train_batches(tr_dl, model, loss_fn, optimizer,
                             log=log, device=device)
        loss_train_time.append(loss)
        loss_val = validate(v_dl, model, loss_fn, log=log,
                            device=device)
        loss_val_time.append(loss_val)
        el = time() - t
        print(f"Trained epoch in {el:.1f} sec")
        save_best_model(model, loss_val_time, title=title)
        i += 1
    print('DONE!')
    torch.save(model.state_dict(), f'./models/{title}_last.pt')
    return model, loss_train_time, loss_val_time


def train_cross(tr_dls, v_dls, model, loss_fn, optimizer_fn, log=0,
                title='flim', epochs=EPOCHS, device=DEVICE):
    models = []
    loss_train_time_n = []
    loss_val_time_n = []
    for k in tr_dls:
        model = load_model(title, model, name='weights')
        optimizer = optimizer_fn(model)
        m, ltt, lvt = train_epochs(tr_dls[k], v_dls[k], model, loss_fn,
                                   optimizer, log, f'{title}/{k}', epochs,
                                   device)
        models.append(m)
        loss_train_time_n.append(ltt)
        loss_val_time_n.append(lvt)
    return models, loss_train_time_n, loss_val_time_n


def train(cross, *args, **kwargs):
    if cross:
        return train_cross(*args, **kwargs)
    else:
        return train_epochs(*args, **kwargs)


# run the testing step
def test(dataloader, model, device=DEVICE):
    model.eval()
    model.to(device)
    dataloader.dataset.test()
    res = []
    gt = []
    with torch.no_grad():
        for i, (data, y, _, _) in tqdm(enumerate(dataloader),
                                       total=len(dataloader)):
            data = data.to(device)
            pred = model(data)
            res.extend(pred.detach().cpu())
            gt.extend(y.detach().cpu())
    res = torch.stack(res)
    gt = torch.stack(gt)
    return res, gt
