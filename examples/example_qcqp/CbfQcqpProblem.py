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

from QcqpProblem import QcqpProblem
from rayen import utils

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

torch.set_default_dtype(torch.float64)


class CbfQcqpProblem(QcqpProblem):
    def __init__(self, X, xo_dim, xc_dim, y_dim, valid_frac=0.0833, test_frac=0.0833):
        super().__init__(X, xo_dim, xc_dim, y_dim, valid_frac, test_frac)
        self.num_cstr = [0, 0, 1, 0, 0]  # just linear constraints

    def objInputMap(self, xo):
        # xo is 3x1
        P = torch.eye(self.xo_dim, self.xo_dim)
        q = -xo
        return P, q

    def getBoundConstraint(self, xc):
        # All 2D tensor
        dim = self.y_dim
        P = torch.eye(dim, dim)
        P_sqrt = P
        q = torch.zeros(dim, 1)
        u_limit = 1.0  # CHANGE
        r = -torch.tensor([[u_limit**2]])
        return P, P_sqrt, q, r

    def getCbfConstraint(self, xc):
        # All 2D tensor
        # xc size (dim*2, 1) including 3D pos and vel
        dim = self.y_dim
        alpha1 = 1.0
        vel_limit = torch.tensor(1.0)  # CHANGE
        t1 = torch.eye(dim, dim)
        vel_select = torch.cat((torch.zeros(dim, dim), t1), dim=1)
        vel = vel_select @ xc
        A = 2 * vel.transpose(-1, -2)
        b = alpha1 * (vel_limit**2 - vel.transpose(-1, -2) @ vel)
        return A, b

    def cstrInputMap(self, xc):
        A2, b2 = utils.getEmpty(), utils.getEmpty()
        A1, b1 = self.getCbfConstraint(xc)
        P, P_sqrt, q, r = self.getBoundConstraint(xc)
        M, s, c, d = utils.getNoneSocConstraints()
        F = utils.getNoneLmiConstraints()
        return A1, b1, A2, b2, P, P_sqrt, q, r, M, s, c, d, F
