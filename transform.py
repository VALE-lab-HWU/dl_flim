import numpy as np
# import torch
from torchvision import transforms

LOG = False


class Flip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        res = sample
        if np.random.rand() < self.flip_prob:
            if np.random.rand() > 0.5:  # hflip
                if LOG:
                    print('flip horizontal')
                res = [transforms.functional.hflip(i).clone() for i in res]
            else:  # vflip
                if LOG:
                    print('flip vertical')
                res = [transforms.functional.vflip(i).clone() for i in res]
        else:
            if LOG:
                print('flip false')
        return res


class Rotate(object):
    def __init__(self, angle, fill=False):
        self.angle = angle
        self.fill = fill

    def __call__(self, sample):
        angle = np.random.uniform(-self.angle, self.angle)
        if LOG:
            print('rotate angle', angle)
        if self.fill:
            res = [transforms.functional.rotate(
                i, angle, fill=float(i[0][0][0])) for i in sample]
        else:
            res = [transforms.functional.rotate(i, angle) for i in sample]
        return res


def get_transforms(angle=None, flip_prob=None, fill_r=False):
    transform_list = []
    if angle is not None:
        transform_list.append(Rotate(angle, fill=fill_r))
    if flip_prob is not None:
        transform_list.append(Flip(flip_prob))
    return transforms.Compose(transform_list)
