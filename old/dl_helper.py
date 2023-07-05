import torch
from torchvision.models import resnet18
from torchvision.models import resnext50_32x4d
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score

from tqdm import tqdm_notebook as tqdm
from time import time

EPOCHS = 10
NUM_CLASSES = 2
BATCH_SIZE = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# evaluate prediction
def evaluate_pred(y_test, pred):
    y_pred = pred.argmax(axis=1)
    return accuracy_score(y_test, y_pred)


# compute loss for the validation dataset
def validate(val_dl, model, loss_fn, log=0, device=DEVICE):
    model.to(device)
    model.eval()
    loss = 0
    with torch.no_grad():
        for _, (data, labels) in tqdm(enumerate(val_dl), total=len(val_dl)):
            data = data.to(device)
            labels = labels.to(device)
            pred = model(data)
            l = loss_fn(pred, labels)
            loss += l.item() / len(val_dl)
        if log == 1:
            print(f"validate loss: {loss:>7f}")
    return loss


# train and test
# one step of training
def train_step(X, y, model, loss_fn, optimizer):
    pred = model(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


# do one epoch
def train_batches(train_dataloader, model, loss_fn, optimizer, log=0,
                  device=DEVICE):
    size = len(train_dataloader.dataset)
    n_batch = len(train_dataloader)
    for batch, (X, y) in tqdm(enumerate(train_dataloader), total=n_batch):
        loss = train_step(X.to(device).float(), y.to(device), model,
                          loss_fn, optimizer)
        if log == 2:
            if batch % (n_batch//10) == 0:
                l, current = loss.item(), batch * len(X)
                print(f"training loss: {l:>7f}  [{current}/{size}]")
    if log == 1:
        l = loss.item()
        print(f"training loss: {l:>7f}")
    return loss


# run through epoch
def train_epochs(train_dl, val_dl, model, loss_fn, optimizer, log=0,
                 epochs=EPOCHS, device=DEVICE):
    val_loss = validate(val_dl, model, loss_fn, log=log, device=device)
    for i in range(epochs):
        t = time()
        print(f"Epoch {i+1}\n-------------------------------")
        loss = train_batches(train_dl, model, loss_fn, optimizer, log=log,
                             device=device)
        val_loss = validate(val_dl, model, loss_fn, log=log, device=device)
        el = time() - t
        print(f"Train epoch in {el:.1f} sec")
    print('DONE!')
    print(loss.item())
    return model


# run the testing step
def test(test_dl, model, device=DEVICE):
    model.to(device)
    model.eval()
    res = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(test_dl), total=len(test_dl)):
            data = data.to(device)
            pred = model(data)
            res.extend(pred.tolist())
    return torch.tensor(res)
