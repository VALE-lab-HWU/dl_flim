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


# save best model
def save_best_model(model, loss, title='unet'):
    if len(loss) <= 2 or (len(loss) > 2 and loss[-1] < min(loss[:-1])):
        print('saving model')
        torch.save(model.state_dict(), f'./models/{title}.pt')


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
def train_epochs(tr_dl, v_dl, model, loss_fn, optimizer, log=0, title='unet',
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
    return model, loss_train_time, loss_val_time


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
