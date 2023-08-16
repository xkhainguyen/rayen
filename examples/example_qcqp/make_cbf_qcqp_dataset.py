# --------------------------------------------------------------------------
# Khai Nguyen | xkhai@cmu.edu
# Robotic Systems Lab, ETH Zürich
# Robotic Exploration Lab, Carnegie Mellon University
# See LICENSE file for the license information
# --------------------------------------------------------------------------

# Code inspired https://github.com/locuslab/DC3/blob/main/datasets/simple/make_dataset.py
# This class stores the problem data and solve optimization

import numpy as np
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader

import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))

from CbfQcqpProblem import CbfQcqpProblem
from rayen import utils

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

torch.set_default_dtype(torch.float64)


def check_balance(data, dataset):
    y_dim = data.y_dim
    count_filtered = 0
    count_nonfiltered = 0
    for i in range(len(dataset)):
        X, Y = dataset[i]
        for j in range(y_dim):
            if torch.norm(X[j] - Y[j]) <= 1e-5:
                count_nonfiltered = count_nonfiltered + 1
            else:
                count_filtered = count_filtered + 1
    print(f"{count_filtered=}; {count_nonfiltered=}")


if __name__ == "__main__":
    num_samples = 200
    xo_dim = 2  # nominal control u_bar dimension
    y_dim = 2  # filtered control, output of the network
    pos_dim = 2
    vel_dim = 2
    xc_dim = pos_dim + vel_dim  # state x dimension

    np.random.seed(1999)
    # should normalize all input
    Xo = np.random.uniform(-1, 1.0, size=(num_samples, xo_dim))
    Xc_pos = np.random.uniform(-1, 1.0, size=(num_samples, pos_dim))
    Xc_vel = np.random.uniform(-1, 1.0, size=(num_samples, vel_dim))
    X = np.hstack((Xo, Xc_pos, Xc_vel))

    problem = CbfQcqpProblem(X, xo_dim, xc_dim, y_dim, valid_frac=0.1, test_frac=0.1)

    # Augment the intact samples
    # for i in range(num_samples):
    #     # for i in range(2):
    #     xc = X[i : i + 1, xo_dim:]
    #     for j in range(20):
    #         xoj = np.random.uniform(-1, 1.0, size=(1, xo_dim))
    #         X = np.vstack((X, np.hstack([xoj, xc])))

    for i in range(int(num_samples)):
        # for i in range(2):
        xc = X[i : i + 1, xo_dim:].T
        A1, b1, *_ = problem.cstrInputMap(torch.tensor(xc))
        for j in range(2):
            xoj = np.random.uniform(-1, 1.0, size=(xo_dim, 1))
            if torch.all((A1) @ torch.tensor(xoj) <= (b1)):
                # print(xoj)
                X = np.vstack([X, np.vstack([xoj, xc]).T])
    np.random.shuffle(X)
    problem = CbfQcqpProblem(X, xo_dim, xc_dim, y_dim, valid_frac=0.1, test_frac=0.1)

    problem.updateObjective()
    problem.updateConstraints()
    problem.calc_Y()
    print(f"{len(problem.Y)=}")

    print(f"{problem.train_num=}; {problem.valid_num=}; {problem.test_num=}")

    data = problem

    # All data tensor
    dataset = TensorDataset(data.X, data.Y)
    train_dataset = torch.utils.data.Subset(dataset, range(data.train_num))
    valid_dataset = torch.utils.data.Subset(
        dataset, range(data.train_num, data.train_num + data.valid_num)
    )
    test_dataset = torch.utils.data.Subset(
        dataset,
        range(
            data.train_num + data.valid_num,
            data.train_num + data.valid_num + data.test_num,
        ),
    )

    check_balance(data, train_dataset)
    check_balance(data, valid_dataset)
    check_balance(data, test_dataset)

    print(train_dataset[0])
    print(train_dataset[2])
    print(problem.obj_val)

    if not os.path.exists("./data"):
        os.makedirs("./data")
    with open(
        "./data/cbf_qcqp_dataset_xo{}_xc{}_ex{}".format(
            xo_dim, xc_dim, problem.nsamples
        ),
        "wb",
    ) as f:
        pickle.dump(problem, f)
