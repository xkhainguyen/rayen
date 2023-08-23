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

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_dtype(torch.float64)

with open(
    "/home/khai/SSD/Code/rayen/examples/example_qcqp2/results/QcqpProblem-2-4-8040/Aug23_11-29-31/args.dict",
    "rb",
) as f:
    data = pickle.load(f)

print(data)
