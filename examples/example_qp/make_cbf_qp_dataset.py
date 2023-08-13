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

import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))

from CbfQpProblem import CbfQpProblem
from rayen import utils

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

torch.set_default_dtype(torch.float64)

if __name__ == "__main__":
    num_samples = 50000
    xo_dim = 1  # nominal control u_bar dimension
    y_dim = 1  # filtered control, output of the network
    pos_dim = 1
    vel_dim = 1
    xc_dim = pos_dim + vel_dim  # state x dimension

    np.random.seed(12)
    # should normalize all input
    Xo = np.random.uniform(-1, 1.0, size=(num_samples, xo_dim))
    Xc_pos = np.random.uniform(-1, 1.0, size=(num_samples, pos_dim))
    Xc_vel = np.random.uniform(-0.9, 0.9, size=(num_samples, vel_dim))
    X = np.hstack((Xo, Xc_pos, Xc_vel))

    problem = CbfQpProblem(X, xo_dim, xc_dim, y_dim, valid_frac=0.1, test_frac=0.1)
    problem.updateObjective()
    problem.updateConstraints()
    problem.calc_Y()
    print(len(problem.Y))

    with open(
        "./cbf_qp_dataset_xo{}_xc{}_ex{}".format(xo_dim, xc_dim, problem.nsamples),
        "wb",
    ) as f:
        pickle.dump(problem, f)
