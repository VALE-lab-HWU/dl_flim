import os
import sys

sys.path.append(os.path.dirname(os.path.abspath('.'))+'/lime')

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import dl_helper as dlh
import torch
import numpy as np
import ml_helper as mlh
import data_helper as dh
from PIL import Image
from sklearn.metrics import accuracy_score
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.measure import block_reduce


def get_hyperparameter(model):
    loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate = 1e-3
    lambda_l2 = 1e-5
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=lambda_l2)  # built-in L2
    return loss_fn, optimizer


def one():
    """
    data from the Labeled Faces in the Wild (lfw) dataset
    """
    fw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    data = fw_people.images.reshape(len(fw_people.data), 1, 50, 37)
    label = fw_people.target
    x_train, x_test, y_train, y_test = train_test_split(
        data, label)
    layer_1 = dlh.create_default_2d_layer()
    pred = dlh.train_and_test(x_train, y_train, x_test, layer_1,
                              len(fw_people.target_names),
                              get_hyperparameter, dlh.DEVICE, epochs=20)
    print(accuracy_score(y_test, pred))


def read_histo_from_path(path):
    print(f"read from {path}")
    names = []
    labels = []
    for i in os.listdir(path):
        tmp = os.listdir(path+'/'+i)
        names.extend(tmp)
        labels.extend([i] * len(tmp))
    data = []
    new_labels = []
    for i, name in enumerate(names):
        print(f"{i+1}/{len(names)}      ", end="\r")
        read = np.array(Image.open(f"{path}/{labels[i]}/{name}"))
        if read.shape == (50, 50, 3):
            data.append(rgb2gray(read))
            new_labels.append(int(labels[i][0]))
    data = np.array(data)
    labels = np.array(new_labels)
    return data, labels


def read_histo_small():
    path = "../data/Breast Histopathology Images SMALL"
    train_path = path+'/train'
    test_path = path+'/test'
    x_train, y_train = read_histo_from_path(train_path)
    x_test, y_test = read_histo_from_path(test_path)
    return x_train, y_train, x_test, y_test


def reshape(x):
    return x.reshape(x.shape[0], 1, *x.shape[1:])


def two():
    """
    data from Breast Histopathology Images dataset on Kaggle
    https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
    """
    x_train, y_train, x_test, y_test = read_histo_small()
    n_class = len(np.unique(y_train))
    layer_1 = dlh.create_default_2d_layer()
    x_train = reshape(x_train)
    x_test = reshape(x_test)
    pred, model = dlh.train_and_test(x_train, y_train, x_test, layer_1,
                                     n_class,
                                     get_hyperparameter, dlh.DEVICE, epochs=20)
    print(accuracy_score(y_test, pred))
    return model


def three():
    """
    data from my own PhD research
    intensity
    """
    lf, label, patient, band = dh.get_data_complete(dh.PATH_CLEANED,
                                                    dh.FILENAME, feature='lf')
    lf = lf.reshape(-1, 1, 128, 128)
    x_train, y_train, x_test, y_test = mlh.split_on_patients(lf, label,
                                                             patient, split=3)
    n_class = len(np.unique(y_train))
    layer_1 = dlh.create_default_2d_layer()
    pred = dlh.train_and_test(x_train, y_train, x_train, layer_1,
                              n_class,
                              get_hyperparameter, dlh.DEVICE, epochs=20)
    print(accuracy_score(y_test, pred))



######################
import torchvision.transforms.functional as TF


def read_histo_from_path(path):
    print(f"read from {path}")
    names = []
    labels = []
    for i in os.listdir(path):
        tmp = os.listdir(path+'/'+i)
        names.extend(tmp)
        labels.extend([i] * len(tmp))
    data = []
    new_labels = []
    for i, name in enumerate(names):
        print(f"{i+1}/{len(names)}      ", end="\r")
        read = Image.open(f"{path}/{labels[i]}/{name}")
        if read.size == (50, 50) and read.mode == 'RGB':
            # data.append(rgb2gray(read))
            tmp = TF.to_tensor(read)
            data.append(tmp.reshape(1, *tmp.shape))
            new_labels.append(int(labels[i][0]))
    data = torch.cat(data)
    labels = torch.tensor(new_labels)
    print('')
    return data, labels

def read_img(path='/train'):
    path = "../data/Breast Histopathology Images SMALL"+path
    x, y = read_histo_from_path(path)
    # return reshape(x), y
    return x, y


def read_histo_small():
    path = "../data/Breast Histopathology Images SMALL"
    train_path = path+'/train'
    val_path = path+'/valid'
    test_path = path+'/test'
    x_train, y_train = read_histo_from_path(train_path)
    x_val, y_val = read_histo_from_path(val_path)
    x_test, y_test = read_histo_from_path(test_path)
    # return reshape(x_train), y_train, reshape(x_val), y_val,reshape(x_test), y_test
    return x_train, y_train, x_val, y_val, x_test, y_test


def reshape(x):
    return x.permute(0, 3, 2, 1)


x_train, y_train = read_img('/train') 
x_val, y_val = read_img('/valid')
x_test, y_test = read_img('/test')





path = "../data/Breast Histopathology Images SMALL/train"
path2 = "../data/Breast Histopathology Images SMALL/test"
path3 = "../data/Breast Histopathology Images SMALL/valid"
print(f"read from {path}")
# names = []
# labels = []
# for i in os.listdir(path):
#     tmp = os.listdir(path+'/'+i)
#     names.extend(tmp)
#     labels.extend([i] * len(tmp))
# data = []
# new_labels = []

# n = np.array([i.split('_') for i in names])
# n = n[:, [0, 2, 3]]
# n[:, 1] = [int(i.replace('x', '')) for i in n[:, 1]]
# n[:, 2] = [int(i.replace('y', '')) for i in n[:, 2]]
# n = n.astype(int)


# u = np.unique(n[:, 0])
# inn = [n[:, 0] == i for i in u]
# nn = [n[i] for i in inn]
# names = np.array(names)
# labels = np.array(labels)
# names2 = [names[i] for i in inn]
# labels2 = [labels[i] for i in inn]


def read_one(path):
    names = []
    labels = []
    for i in os.listdir(path):
        tmp = os.listdir(path+'/'+i)
        names.extend(tmp)
        labels.extend([i] * len(tmp))


        n = np.array([i.split('_') for i in names])
    n = n[:, [0, 2, 3]]
    n[:, 1] = [int(i.replace('x', '')) for i in n[:, 1]]
    n[:, 2] = [int(i.replace('y', '')) for i in n[:, 2]]
    n = n.astype(int)
    u = np.unique(n[:, 0])
    inn = {i: n[:, 0] == i for i in u}
    nn = {i: n[inn[i]] for i in inn}
    names = np.array(names)
    labels = np.array(labels)
    names2 = {i: names[inn[i]] for i in inn}
    labels2 = {i: labels[inn[i]] for i in inn}
    return nn, names2, labels2

nn, names, labels = read_one(path)
nn2, names2, labels2 = read_one(path2)
nn3, names3, labels3 = read_one(path3)

    
idx = 8863
p = nn[idx]
p2 = nn2[idx]
p3 = nn3[idx]
_, xmax, ymax = p.max(axis=0)
_, xmax2, ymax2 = p.max(axis=0)
_, xmax3, ymax3 = p.max(axis=0)


fig, ax = plt.subplots()
ax.set_xlim(0, max(xmax, xmax2, xmax3))
ax.set_ylim(0, max(ymax, ymax2, ymax3))


def add_to_grid(p, ax, labels, names, idx, path):
    for i in range(len(p)):
        a1 = Image.open(f"{path}/{labels[idx][i]}/{names[idx][i]}")
        ax.imshow(a1, extent=(p[i][1], p[i][1]+a1.width,
                              p[i][2], p[i][2]-a1.height))

print('0')
add_to_grid(p, ax, labels, names, idx, path)
print('2')
add_to_grid(p2, ax, labels2, names2, idx, path2)
print('3')
add_to_grid(p3, ax, labels3, names3, idx, path3)


# i = np.argsort(n[:, 2])
# n = n[i]
# i2 = np.argsort(n[:, 1], kind='mergesort')
# n = n[i2]
# names = np.array(names)[i][i2]
# labels = np.array(labels)[i][i2]

# fig, ax = plt.subplots(2)
# for j in [1, 2]:
#     a1 = Image.open(f"{path}/{labels[j]}/{names[j]}")
#     ax[j-1].imshow(a1)
# ## groulp by n[0] first
def add_to_grid(p, ax, names, path, c):
    for i in range(len(p)):
        img = Image.open(f"{path}/{names[i]}")
        ax.imshow(img, extent=(p[i][1], p[i][1]+img.width,
                               p[i][2], p[i][2]-img.height))
        ax.imshow(c, extent=(p[i][1], p[i][1]+img.width,
                             p[i][2], p[i][2]-img.height))


def process_names(names):
    n = np.array([i.split('_') for i in names])
    n = n[:, [0, 2, 3]]
    n[:, 1] = [int(i.replace('x', '')) for i in n[:, 1]]
    n[:, 2] = [int(i.replace('y', '')) for i in n[:, 2]]
    n = n.astype(int)
    return n


def read_one_label(path, ax, c):
    names = os.listdir(path)
    names = np.array(names)
    n = process_names(names)
    _, xmax, ymax = n.max(axis=0)
    add_to_grid(n, ax, names, path, c)
    return xmax, ymax


def read_one_idx(path, idx, ax):
    c0 = [[[0, 0, 1, 0.1]]]
    c1 = [[[0, 1, 0, 0.1]]]
    xmax0, ymax0 = read_one_label(f'{path}/{idx}/0', ax, c0)
    xmax1, ymax1 = read_one_label(f'{path}/{idx}/1', ax, c1)
    ax.set_xlim(0, max(xmax0, xmax1)+50)
    ax.set_ylim(0, max(ymax0, ymax1)+50)


fig, ax = plt.subplots()

path = "../data/Breast Histopathology Images"
listdir = os.listdir(path)
read_one_idx(path, listdir[0], ax)

# path2 = path + '/IDC_regular_ps50_idx5'
# read_one_idx(path2, listdir[0], ax)


def reconstruct_one_label(n, names, img, path, grid, label):
    for i in range(len(n)):
        read = Image.open(f"{path}/{names[i]}")
        # print(img.shape)
        # print('n', n[i], 'h', read.height, 'w', read.width)
        # print(n[i][2],n[i][2]+read.height)
        # print(n[i][1],n[i][1]+read.width)
        img[n[i][2]:n[i][2]+read.height,
            n[i][1]:n[i][1]+read.width] = np.array(read)
        grid[n[i][2]:n[i][2]+read.height,
             n[i][1]:n[i][1]+read.width] = label
    return img


def names_one_label(path):
    names = os.listdir(path)
    names = np.array(names)
    n = process_names(names)
    _, xmax, ymax = n.max(axis=0)
    return n, names, xmax, ymax


def reconstruct_image(path, idx):
    path0 = f'{path}/{idx}/0'
    path1 = f'{path}/{idx}/1'
    n0, names0, xmax0, ymax0 = names_one_label(
        path0)
    n1, names1, xmax1, ymax1 = names_one_label(
        path1)
    img = np.full((max(ymax0, ymax1)+50, max(xmax0, xmax1)+50, 3), 0)
    grid = np.full((max(ymax0, ymax1)+50, max(xmax0, xmax1)+50), 0)
    img = reconstruct_one_label(n0, names0, img, path0, grid, 1)
    img = reconstruct_one_label(n1, names1, img, path1, grid, 2)
    return img.astype(np.uint8), grid.astype(np.int8)


def reconstruct_images(path):
    listdir = os.listdir(path)
    for i in range(len(listdir)):
        print(f'{i+1}/{len(listdir)}', end='\r')
        try:
            int(listdir[i])
        except ValueError:
            continue
        img, grid = reconstruct_image(path, listdir[i])
        Image.fromarray(img).save(f'{path}/{listdir[i]}.png')
        Image.fromarray(grid).save(f'{path}/{listdir[i]}_grid.png')



path = "../data/Breast Histopathology Images"
reconstruct_images(path)


####
####
def take_nxm(i, j, n, m, grid):
    return grid[i:i+n, j:j+m]


def create_nxm_from_one_image(n, m, path, name):
    img = np.array(Image.open(f"{path}/{name}.png"))
    grid = np.array(Image.open(f"{path}/{name}_grid.png"))
    grid_8 = block_reduce(grid, (8, 8), np.min)
    n8 = round(n/8)
    m8 = round(m/8)
    i_max = len(grid_8) - n8
    j_max = len(grid_8[0]) - m8
    for _ in range(500):
        i, j = np.random.randint((i_max, j_max))
        tmp_grid_8 = take_nxm(i, j, n8, m8, grid_8)
        if not (tmp_grid_8 == 0).any():
            i = i*8
            j = j*8
            print(f'found in {i} {j}')
            tmp_grid = take_nxm(i, j, n, m, grid)
            if (tmp_grid == 2).sum() > (n*m//10):
                label = 2
            else:
                label = 1
            tmp_img = take_nxm(i, j, n, m, img)
            Image.fromarray(tmp_img) \
                 .save(f'{path}/data_128/{label}/{name}_{i}_{j}.png')


def create_nxm_images(n, m, path):
    listdir = os.listdir(path)
    for i in range(len(listdir)):
        print(f'{i+1}/{len(listdir)}', end='\r')
        try:
            int(listdir[i])
        except ValueError:
            continue
        create_nxm_from_one_image(n, m, path, listdir[i])


path = "../data/Breast Histopathology Images"
create_nxm_images(128, 128, path)
