import os
import torch
import random
import warnings
import numpy as np
from string import Formatter
from datetime import datetime, timedelta

basedir = './'

def timmer(fn):
    def wrapper(*args, **kwargs):
        a = datetime.now()
        res = fn(*args, **kwargs)
        b = datetime.now()
        print('Func:', fn.__name__, ', Time usage: ', strfdelta(b - a))
        return res

    return wrapper


# model loss
class LOSS_MODE():
    NO_LOSS = 0
    BERT_LOSS = 1
    CLIP_LOSS = 2
    NCCL_LOSS = 3
    SYMI_LOSS = 4


# for model wrapper input
class INPUT_MODE():
    CODE = 0
    NL = 1


# generate dataset
class DATA_MODE():
    TRAIN = 0
    CODE = 1
    QUERY = 2


# evaluate
class EVAL_MODE():
    TEST = 0
    VALID = 1


# set tensor tuple or list to device
def to_device(v, device):
    if isinstance(v, tuple):
        return tuple(to_device(x, device) for x in v)
    elif isinstance(v, list):
        return [to_device(x, device) for x in v]
    else:
        return v.to(device)


def set_seed(seed=43):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')


def strfdelta(tdelta, fmt='{D:2}d {H:2}:{M:02}:{S:02}', inputtype='timedelta'):
    if inputtype == 'timedelta':
        remainder = int(tdelta.total_seconds())
    elif inputtype in ['s', 'seconds']:
        remainder = int(tdelta)
    elif inputtype in ['m', 'minutes']:
        remainder = int(tdelta) * 60
    elif inputtype in ['h', 'hours']:
        remainder = int(tdelta) * 3600
    elif inputtype in ['d', 'days']:
        remainder = int(tdelta) * 86400
    elif inputtype in ['w', 'weeks']:
        remainder = int(tdelta) * 604800

    f = Formatter()
    desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
    possible_fields = ('W', 'D', 'H', 'M', 'S')
    constants = {'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
    values = {}
    for field in possible_fields:
        if field in desired_fields and field in constants:
            values[field], remainder = divmod(remainder, constants[field])
    return f.format(fmt, **values)
