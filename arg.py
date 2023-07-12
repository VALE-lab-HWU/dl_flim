import argparse

def parse_args_ds(argp):
    argp.add_argument('--n_img', type=int, default=-1,
                      help="Limit to the number of image to load",
                      dest='ds_n_img')
    return argp


def parse_args_dl(argp):
    argp.add_argument("--shuffle", action='store_true',
                      help="Flag to not shuffle when traning (for epochs)",
                      dest='dl_shuffle')
    argp.set_defaults(dl_shuffle=False)
    argp.add_argument("--no_split_shuffle", action='store_false',
                      help="Flag to not shuffle when splitting",
                      dest='dl_split_shuffle')
    argp.set_defaults(dl_split_shuffle=True)
    argp.add_argument('--batch_size', type=int, default=1,
                      help="The bath size to compute the loss on",
                      dest='dl_batch_size')
    argp.add_argument('--test_subset', type=float,
                      help="Percentage size for the training/testing split",
                      dest='dl_test_subset')
    argp.add_argument('--test_patient', type=str,
                      help="Patient to use for testing",
                      dest='dl_test_subset')
    argp.set_defaults(dl_test_subset=0.3)
    argp.add_argument('--val_subset', type=float,
                      help="Percentage size for the training/validation split",
                      dest='dl_val_subset')
    argp.add_argument('--val_patient', type=str,
                      help="Patient to use for validation",
                      dest='dl_val_subset')
    argp.set_defaults(dl_val_subset=0.3)

    return argp


def parse_args_md(argp):
    argp.add_argument('--epochs', type=int, default=1000,
                      help="The number of epochs to train the model",
                      dest='md_epochs')
    argp.add_argument('--learning_rate', type=float, default=1e-4,
                      help="The learning rate to which train the model",
                      dest='md_learning_rate')
    return argp


def parse_args_tf(argp):
    argp.add_argument('--angle', type=int,
                      help="Angle value for data augmentation",
                      dest='tf_angle')
    argp.add_argument('--no_angle', help='no angle', dest='tf_angle',
                      action='store_const', const=None)
    argp.set_defaults(tf_angle=180)
    argp.add_argument('--flip', type=float,
                      help="Flip probability value for data augmentation",
                      dest='tf_flip')
    argp.add_argument('--no_flip', help='no flip', dest='tf_flip',
                      action='store_const', const=None)
    return argp


def choose_gpu(arg):
    if arg == 'cuda':
        return arg
    elif arg in ['0', '1', '2', '3']:
        return f'cuda:{arg}'
    else:
        raise argparse.ArgumentTypeError('GPU should be 0, 1, 2, or 3')


def parse_args(name):
    argp = argparse.ArgumentParser(name)
    argp.add_argument("--k_cross", action='store_true',
                      help="Flag to add K-foldcross validation")
    argp.set_defaults(k_cross=False)
    argp.add_argument("--p_cross", action='store_true',
                      help="Flag to add cross validation per patient")
    argp.set_defaults(p_cross=False)
    argp.add_argument("--cross_nb", type=int, default=5,
                      help="Integer representing how many fold for CV")
    argp.add_argument('--device', default='cuda', type=choose_gpu,
                      help="Which gpu to use. Default is all")
    argp.add_argument('--seed', type=int, default=42,
                      help="Percentage size for the training/validation split")
    argp.add_argument('--log', choices=range(2), type=int,
                      default=1,
                      help="Log level. Can be 0 (nothing) or 1-2")
    argp.add_argument('--title', type=str, default="flim_cnn",
                      help="Title of the file to save the model in")
    argp = parse_args_ds(argp)
    argp = parse_args_dl(argp)
    argp = parse_args_md(argp)
    argp = parse_args_tf(argp)
    return argp.parse_args()
