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

from CbfSocProblem import CbfSocProblem
from rayen import utils, constraint_module, constraints_torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

DEVICE = torch.device("cpu")
torch.set_default_device(DEVICE)
torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=4)

seed = 1999
torch.manual_seed(seed)
np.random.seed(seed)


def check_balance(data, dataset):
    y_dim = data.y_dim
    count_filtered = 0
    count_nonfiltered = 0
    for i in range(len(dataset)):
        X, Y = dataset[i]
        for j in range(y_dim):
            if torch.norm(X[j] - Y[j]) <= 1e-2:
                count_nonfiltered = count_nonfiltered + 1
            else:
                count_filtered = count_filtered + 1
    print(f"{count_filtered=}; {count_nonfiltered=}")


def filter_samples(data, cond):
    # input X is a 2D np array, cond is a condition function
    n = len(data)
    data_n = []
    for i in range(n):
        if cond(data[i]):
            data_n.append(data[i])
        else:
            data_n.append(np.ones(data[i].shape) * np.nan)
    data_nn = np.array(data_n)
    mask = ~np.isnan(data_nn).all(axis=1)
    return data[mask]


if __name__ == "__main__":
    utils.printInBoldBlue("Generating dataset")
    num_samples = 15000
    xo_dim = 3  # nominal control u_bar dimension
    y_dim = 3  # filtered control, output of the network
    pos_dim = 3
    vel_dim = 3
    xc_dim = pos_dim + vel_dim  # state x dimension

    np.random.seed(1999)
    # should normalize all input
    Xo = np.random.uniform(-1.0, 1.0, size=(num_samples, xo_dim))
    Xc_pos = np.random.uniform(-1, 1.0, size=(num_samples, pos_dim))

    Xc_vel = np.random.uniform(-0.6, 0.6, size=(num_samples, vel_dim))
    X_raw = np.hstack((Xo, Xc_pos, Xc_vel))

    problem_raw = CbfSocProblem(
        X_raw, xo_dim, xc_dim, y_dim, valid_frac=0.1, test_frac=0.1
    )

    for i in range(int(num_samples)):
        # for i in range(2):
        xc = X_raw[i : i + 1, xo_dim:].T
        A1, b1, _, _, P, q, r, M, s, c, d, *_ = problem_raw.cstrInputMap(
            torch.tensor(xc)
        )
        for j in range(100):
            xoj12 = np.random.uniform(-0.6, 0.6, size=(xo_dim - 1, 1))
            xoj3 = np.random.uniform(0, 0.6, size=(1, 1))
            xoj = np.vstack((xoj12, xoj3))
            # print(xoj)
            Xoj = torch.tensor(xoj)
            cond1 = torch.all(A1 @ Xoj <= b1)
            cond2 = torch.all(
                0.5 * Xoj.transpose(-1, -2) @ P @ Xoj + q.transpose(-1, -2) @ Xoj + r
                <= 0.0
            )
            cond3 = torch.all(
                torch.norm(M * Xoj + s) - c.transpose(-1, -2) @ Xoj - d <= 0
            )
            if cond1 and cond2 and cond3:
                X_raw = np.vstack([X_raw, np.vstack([xoj, xc]).T])

    X = filter_samples(X_raw, lambda x: x[6] ** 2 + x[7] ** 2 + x[8] ** 2 <= 1.0 - 1e-2)

    np.random.shuffle(X)

    problem = CbfSocProblem(X, xo_dim, xc_dim, y_dim, valid_frac=0.1, test_frac=0.1)
    problem.updateConstraints()

    layer = constraint_module.ConstraintModule(
        problem.y_dim,
        problem.xc_dim,
        problem.y_dim,
        "RAYEN",
        problem.num_cstr,
        problem.cstrInputMap,
    )

    utils.printInBoldBlue("finding solution with numerical solvers")
    problem.updateObjective()
    problem.computeY(tol=1e-8)
    print(f"{len(problem.Y)=}")

    print(f"{problem.train_num=}; {problem.valid_num=}; {problem.test_num=}")

    # # All data tensor
    dataset = TensorDataset(problem.X, problem.Y)
    # print(problem.Y_np)
    # print(problem.X_np)
    check_balance(problem, dataset)

    # Xn = filter_samples(dataset, lambda x: torch.isclose(torch.norm(x[0][0:3], x[1][0:3]), ))
    # for i in range(len(problem.Y)):
    #     print(dataset[i])
    # # print(problem.obj_val)

    utils.printInBoldBlue("finding interior points with cvxpylayers")
    # xv can arbitrary
    # xv can arbitrary
    # Xo = torch.tensor([[1.2, 2.1, -2.3], [1.2, 2.1, -2.4]])
    # Xc = torch.tensor([[0.8, 0.1, 0.2, 0.4, 0.4, 0.4], [0.8, 0.1, 0.2, 0.4, 0.4, 0.5]])
    # Y = layer(Xo, Xc)  # 3D
    Y = layer(problem.Xo.squeeze(-1), problem.Xc.squeeze(-1))  # Y is 3D
    # print(Y)
    problem.updateInteriorPoint(layer.z0)  # 3D
    print(f"{problem.Y0.shape = }; \n{problem.Y0[0] = }")
    print(f"{layer.isFeasible(problem.Y0, 1e-2)}")

    if not os.path.exists("./data"):
        os.makedirs("./data")
    with open(
        "./data/cbf_soc_dataset_xo{}_xc{}_ex{}".format(
            xo_dim, xc_dim, problem.nsamples
        ),
        "wb",
    ) as f:
        pickle.dump(problem, f)
