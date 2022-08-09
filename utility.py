import matplotlib.pyplot as plt

def link_string(*args):
    return '/'.join(args)

def show(im, **kwargs):
    plt.imshow(im, **kwargs)
    plt.show()

def shows(*im):
    fig, axs = plt.subplots(1, len(im))
    for i in range(len(im)):
        axs[i].imshow(im[i])
    plt.show()
