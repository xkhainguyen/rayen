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

# pickle is lazy and does not serialize class definitions or function
# definitions. Instead it saves a reference of how to find the class
# (the module it lives in and its name)
from CbfQpProblem import CbfQpProblem

# Define problem
args = {"probType": "cbf_qp", "xo": 1, "xc": 2, "nsamples": 953}

# Load data, and put on GPU if needed
prob_type = args["probType"]
if prob_type == "cbf_qp":
    filepath = "data/cbf_qp_dataset_xo{}_xc{}_ex{}".format(
        args["xo"], args["xc"], args["nsamples"]
    )
else:
    raise NotImplementedError

with open(filepath, "rb") as f:
    data = pickle.load(f)

for attr in dir(data):
    var = getattr(data, attr)
    if not callable(var) and not attr.startswith("__") and torch.is_tensor(var):
        try:
            setattr(data, attr, var.to(DEVICE))
        except AttributeError:
            pass

data._device = DEVICE

batch_size = 2

dataset = TensorDataset(data.X, data.Y)
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [data.train_num, data.valid_num, data.test_num]
)

# # all tensor
# train_dataset = TensorDataset(data.trainX)
# valid_dataset = TensorDataset(data.validX)
# test_dataset = TensorDataset(data.testX)

# # to batch
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

# print(f"{len(train_dataset)=} {len(valid_dataset)=} {len(test_dataset)=}")
# for X, Y in train_loader:
#     print(X, Y)
#     break


with open(
    "/home/khai/SSD/Code/rayen/examples/example_qp/results/QpProblem-1-2-953/1692015136-4423506/stats.dict",
    "rb",
) as f:
    data = pickle.load(f)

print(data["valid_loss"].shape)
print(np.mean(data["valid_loss"], 1))
print(np.mean(data["valid_loss"][-1]))
