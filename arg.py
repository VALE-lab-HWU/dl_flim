import argparse


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
    argp.add_argument('--test_subset', type=float, default=0.3,
                      help="Percentage size for the training/testing split",
                      dest='dl_test_subset')
    argp.add_argument('--val_subset', type=float, default=0.3,
                      help="Percentage size for the training/validation split",
                      dest='dl_val_subset')
    return argp


def parse_args_md(argp):
    argp.add_argument('--epochs', type=int, default=1000,
                      help="The number of epochs to train the model",
                      dest='md_epochs')
    argp.add_argument('--learning_rate', type=float, default=1e-4,
                      help="The learning rate to which train the model",
                      dest='md_learning_rate')
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
    argp.add_argument('--device', default='cuda', type=choose_gpu,
                      help="Which gpu to use. Default is all")
    argp.add_argument('--seed', type=int, default=42,
                      help="Percentage size for the training/validation split")
    argp.add_argument('--log', choices=range(2), type=int,
                      default=1,
                      help="Log level. Can be 0 (nothing) or 1-2")
    argp = parse_args_dl(argp)
    argp = parse_args_md(argp)
    return argp.parse_args()
