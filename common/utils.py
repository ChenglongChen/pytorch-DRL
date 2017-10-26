
import torch as th
from torch.autograd import Variable
import numpy as np


def entropy(p):
    return -th.sum(p * th.log(p), 1)


def index_to_one_hot(index, dim):
    assert index < dim
    one_hot = np.zeros(dim)
    one_hot[index] = 1
    return one_hot


def to_tensor_var(x, use_cuda=True, dtype="float"):
    FloatTensor = th.cuda.FloatTensor if use_cuda else th.FloatTensor
    LongTensor = th.cuda.LongTensor if use_cuda else th.LongTensor
    ByteTensor = th.cuda.ByteTensor if use_cuda else th.ByteTensor
    if dtype == "float":
        x = np.array(x, dtype=np.float64)
        return Variable(FloatTensor(x))
    elif dtype == "long":
        x = np.array(x, dtype=np.long)
        return Variable(LongTensor(x))
    elif dtype == "byte":
        x = np.array(x, dtype=np.byte)
        return Variable(ByteTensor(x))
    else:
        x = np.array(x, dtype=np.float64)
        return Variable(FloatTensor(x))
