import torch
import torch.nn as nn
import torch.optim as optim

import operator
from functools import reduce
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pickle
import time
import os
import argparse

import sys
from os.path import normpath, dirname, join

sys.path.insert(0, normpath(join(dirname(__file__), "..")))

from CbfSocProblem import CbfSocProblem
from rayen import utils, constraint_module, constraints_torch

DEVICE = torch.device("cpu")
torch.set_default_device(DEVICE)
torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=4)

seed = 1999
torch.manual_seed(seed)
np.random.seed(seed)


# with open(
#     "/home/khai/SSD/Code/rayen/examples/example_qcqp2/results/QcqpProblem-2-4-8040/Aug23_11-29-31/args.dict",
#     "rb",
# ) as f:
#     data = pickle.load(f)

with open(
    "/home/khai/SSD/Code/rayen/examples/example_soc/data/cbf_soc_dataset_xo3_xc6_ex13427",
    "rb",
) as f:
    data = pickle.load(f)
for attr in dir(data):
    var = getattr(data, attr)
    if not callable(var) and not attr.startswith("__") and torch.is_tensor(var):
        try:
            setattr(data, attr, var.to(DEVICE))
        except AttributeError:
            pass
dataset = TensorDataset(data.X, data.Y)
num = 10
for i in range(num):
    print(dataset[i])
