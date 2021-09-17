import torch
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import numpy as np

EPOCHS = 100
NUM_CLASSES = 2
BATCH_SIZE = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# create the nn model
# resnet
def create_model(device=DEVICE):
    # load resnet
    model = resnet18(num_classes=2)
    # adapt first layer to gray
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7),
                                  stride=(2, 2),  padding=(3, 3), bias=False)
    model.to(device)
    return model


# example of hyperparameter function
# should be tuned for each model
def get_hyperparameter(model):
    loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate = 1e-3
    lambda_l2 = 1e-5
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=lambda_l2)  # built-in L2
    return loss_fn, optimizer


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
def train_batches(train_dataloader, model, loss_fn, optimizer, device=DEVICE):
    size = len(train_dataloader.dataset)
    n_batch = len(train_dataloader)
    for batch, (X, y) in enumerate(train_dataloader):
        loss = train_step(X.to(device).float(), y.to(device), model,
                          loss_fn, optimizer)
        if (round(batch % (n_batch/10)) == 0):
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current}/{size}]")
    return loss


# run the testing step
def test(x_test, model, batch_size=BATCH_SIZE, device=DEVICE):
    # necessary? cause of batch size?
    test_dataloader = DataLoader(x_test,
                                 batch_size=batch_size)
    model.eval()
    res = []
    with torch.no_grad():
        for X in test_dataloader:
            X = X.to(device).float()
            pred = model(X)
            res.extend(pred.tolist())
    return res


# loss ?
def compute_loss(pred, y, loss_fn):
    pred = torch.tensor(pred)
    y = torch.tensor(y)
    size = len(pred)
    test_loss, correct = 0, 0
    test_loss += loss_fn(pred, y).item()
    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%,"
          f"Avg loss: {test_loss:>8f} \n")


# run through epoch
def train_epochs(x_train, y_train, model, loss_fn, optimizer,
                 epochs=EPOCHS, batch_size=BATCH_SIZE, device=DEVICE):
    train_dataloader = DataLoader(list(zip(x_train, y_train)),
                                  batch_size=batch_size)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss = train_batches(train_dataloader, model, loss_fn, optimizer,
                             device=device)
    print('DONE!')
    print(loss.item())


def train_and_test(x_train, y_train, x_test,  fn_parameter=get_hyperparameter,
                   device=DEVICE, epochs=EPOCHS, batch_size=BATCH_SIZE):
    model = create_model(device)
    loss_fn, optimizer = fn_parameter(model)
    train_epochs(x_train, y_train, model, loss_fn, optimizer,
                 epochs, batch_size, device)
    pred = test(x_test, model)
    return np.argmax(pred, axis=1)
