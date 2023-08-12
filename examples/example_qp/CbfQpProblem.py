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

from QpProblem import QpProblem
from rayen import utils

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

torch.set_default_dtype(torch.float64)


class CbfQpProblem(QpProblem):
    def __init__(self, X, xo_dim, xc_dim, y_dim, valid_frac=0.0833, test_frac=0.0833):
        super().__init__(X, xo_dim, xc_dim, y_dim, valid_frac, test_frac)
        self.num_cstr = [0, 0, 0, 0, 0]  # just linear constraints

    def objInputMap(self, xo):
        # xo is 3x1
        P = torch.eye(self.xo_dim, self.xo_dim)
        q = -xo
        return P, q

    def getBoundConstraint(self, xc):
        dim = self.y_dim
        t1 = torch.eye(dim, dim)
        t2 = -torch.eye(dim, dim)
        A = torch.cat((t1, t2), dim=0)
        b = torch.ones(dim * 2, 1) * 1.1  # CHANGE
        return A, b

    def getCbfConstraint(self, xc):
        # xc size (6, 1) including 3D pos and vel
        dim = self.y_dim
        alpha1 = 1.0
        alpha2 = 1.0
        pos_box = torch.ones(2 * dim, 1) * 1.1  # CHANGE
        t1 = torch.eye(dim, dim)
        t2 = torch.cat((torch.zeros(dim, dim), t1), dim=1)
        t3 = torch.cat((t1, (torch.zeros(dim, dim))), dim=1)
        vel_select = torch.cat((t2, -t2), dim=0)
        pos_select = torch.cat((t3, -t3), dim=0)
        A = torch.cat((t1, -t1), dim=0)
        b = -(alpha1 + alpha2) * vel_select @ xc + alpha1 * alpha2 * (
            pos_box - pos_select @ xc
        )
        return A, b

    def cstrInputMap(self, xc):
        A2, b2 = utils.getEmpty(), utils.getEmpty()
        A1_bound, b1_bound = self.getBoundConstraint(xc)
        A1_cbf, b1_cbf = self.getCbfConstraint(xc)
        A1 = torch.cat((A1_bound, A1_cbf), dim=0)
        b1 = torch.cat((b1_bound, b1_cbf), dim=0)
        P, P_sqrt, q, r = utils.getNoneQuadraticConstraints()
        M, s, c, d = utils.getNoneSocConstraints()
        F = utils.getNoneLmiConstraints()
        return A1, b1, A2, b2, P, P_sqrt, q, r, M, s, c, d, F
