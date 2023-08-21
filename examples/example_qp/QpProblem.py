# --------------------------------------------------------------------------
# Khai Nguyen | xkhai@cmu.edu
# Robotic Systems Lab, ETH Zürich
# Robotic Exploration Lab, Carnegie Mellon University
# See LICENSE file for the license information
# --------------------------------------------------------------------------

# Code inspired https://github.com/locuslab/DC3/blob/main/utils.py
# This class stores the problem data and solve optimization

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.autograd import Function

torch.set_default_dtype(torch.float64)

import numpy as np
import osqp
import cvxpy as cp
from scipy.linalg import svd
from scipy.sparse import csc_matrix

from copy import deepcopy
import scipy.io as spio
import time
import sys
from os.path import normpath, dirname, join

sys.path.insert(0, normpath(join(dirname(__file__), "../..")))

from rayen import utils

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


###################################################################
# QP PROBLEM
###################################################################


class QpProblem(ABC):
    """
    minimize_y  1/2 * y^T P y + q^Ty
    s.t.        A2y =  b2
                A1y <= b1
    P, q come from objectiveFunction
    A1, b1, A2, b2 come from constraintInpuMap
    """

    def __init__(self, X, xo_dim, xc_dim, y_dim, valid_frac=0.0833, test_frac=0.0833):
        self._X = torch.tensor(X).unsqueeze(-1)
        self._P = None
        self._q = None
        self._A1 = None
        self._b1 = None
        self._Y0 = None
        self._Y = None
        self._obj_val = None
        self._x_dim = X.shape[1]
        self._xo_dim = xo_dim
        self._xc_dim = xc_dim
        assert (xo_dim + xc_dim) == X.shape[1], "input shape is incorrect"
        self._y_dim = y_dim
        self._nsamples = X.shape[0]
        self._nineq = None
        self._neq = None
        self._valid_frac = valid_frac
        self._test_frac = test_frac

        ### For PyTorch
        self._device = None

    def __str__(self):
        return "QpProblem-{}-{}-{}".format(
            str(self.xo_dim), str(self.xc_dim), str(self._nsamples)
        )

    @property
    def x_dim(self):
        return self._x_dim

    @property
    def xo_dim(self):
        return self._xo_dim

    @property
    def xc_dim(self):
        return self._xc_dim

    @property
    def y_dim(self):
        return self._y_dim

    @property
    def nsamples(self):
        return self._nsamples

    @property
    def neq(self):
        return self._neq

    @property
    def nineq(self):
        return self._nineq

    @property
    def P(self):
        return self._P

    @property
    def q(self):
        return self._q

    @property
    def A1(self):
        return self._A1

    @property
    def b1(self):
        return self._b1

    @property
    def X(self):
        return self._X

    @property
    def Xo(self):
        return self.X[:, 0 : self.xo_dim]

    @property
    def Xc(self):
        return self.X[:, self.xo_dim : self.xo_dim + self.xc_dim]

    @property
    def Y0(self):
        return self._Y0

    @property
    def Y(self):
        return self._Y

    @property
    def obj_val(self):
        return self._obj_val

    @property
    def P_np(self):
        return self.P.detach().cpu().numpy()

    @property
    def q_np(self):
        return self.q.detach().cpu().numpy()

    @property
    def A1_np(self):
        return self.A1.detach().cpu().numpy()

    @property
    def b1_np(self):
        return self.b1.detach().cpu().numpy()

    @property
    def X_np(self):
        return self.X.detach().cpu().numpy()

    @property
    def X_np(self):
        return self.X.detach().cpu().numpy()

    @property
    def Xo_np(self):
        return self.Xo.detach().cpu().numpy()

    @property
    def Xc_np(self):
        return self.Xc.detach().cpu().numpy()

    @property
    def Y_np(self):
        return self.Y.detach().cpu().numpy()

    @property
    def Y0_np(self):
        return self.Y0.detach().cpu().numpy()

    @property
    def valid_frac(self):
        return self._valid_frac

    @property
    def test_frac(self):
        return self._test_frac

    @property
    def train_frac(self):
        return 1 - self.valid_frac - self.test_frac

    @property
    def valid_num(self):
        return int(self.nsamples * self.valid_frac)

    @property
    def test_num(self):
        return int(self.nsamples * self.test_frac)

    @property
    def train_num(self):
        return self.nsamples - self.valid_num - self.test_num

    @property
    def device(self):
        return self._device

    @abstractmethod
    def objInputMap(self, xo):
        pass

    @abstractmethod
    def cstrInputMap(self, xc):
        pass

    ## For PyTorch
    def getXo(self, X):
        return X[:, 0 : self.xo_dim]

    def getXc(self, X):
        return X[:, self.xo_dim : self.xo_dim + self.xc_dim]

    def updateObjective(self, Xo=None):
        if Xo is None:
            Xo_ = self.Xo
        else:
            Xo_ = Xo
        self._P, self._q = torch.vmap(self.objInputMap)(Xo_)
        return self._P, self._q

    ## For PyTorch
    def objectiveFunction(self, Y):
        # shape (n, 1)
        return 0.5 * Y.transpose(-1, -2) @ self.P @ Y + self.q.transpose(-1, -2) @ Y

    def updateConstraints(self, Xc=None):
        if Xc is None:
            Xc_ = self.Xc
        else:
            Xc_ = Xc
        self._A1, self._b1, *_ = torch.vmap(self.cstrInputMap)(Xc_)
        return self._A1, self._b1

    def optimizationSolve(self, X, solver_type="cvxpy_osqp", tol=1e-5):
        if solver_type == "osqp":
            utils.printInBoldBlue("running osqp")
            P, q, A1, b1 = (self.P_np, self.q_np, self.A1_np, self.b1_np)
            X_np = X.detach().cpu().numpy()
            Y = []
            obj_val = []
            total_time = 0
            for i in range(self.nsamples):
                solver = osqp.OSQP()

                my_A = A1[i]
                my_l = -np.ones(my_A.shape[0]) * np.inf
                my_u = b1[i]
                solver.setup(
                    P=csc_matrix(P[i]),
                    q=q[i],
                    A=csc_matrix(my_A),
                    l=my_l,
                    u=my_u,
                    verbose=False,
                    max_iter=100,
                    eps_rel=tol,
                    eps_abs=tol,
                )
                start_time = time.time()
                results = solver.solve()
                end_time = time.time()

                total_time += end_time - start_time
                if results.info.status == "solved":
                    Y.append(results.x)
                    obj_val.append(results.info.obj_val)
                else:
                    Y.append(np.ones(self.y_dim) * np.nan)
                    obj_val.append(np.nan)

            # print(Y)
            sols = np.array(Y)
            obj_val = np.array(obj_val)
            parallel_opt_time = total_time / len(X_np)
            print(f"{parallel_opt_time=}")

        elif solver_type == "cvxpy_osqp":
            utils.printInBoldBlue("running cvxpy with osqp")
            P, q, A1, b1 = (self.P_np, self.q_np, self.A1_np, self.b1_np)
            X_np = X.detach().cpu().numpy()
            Y = []
            obj_val = []
            total_time = 0
            for i in range(self.nsamples):
                y = cp.Variable((self.y_dim, 1))
                prob = cp.Problem(
                    cp.Minimize((1 / 2) * cp.quad_form(y, P[i]) + q[i].T @ y),
                    [A1[i] @ y <= b1[i]],
                )

                start_time = time.time()
                prob.solve(
                    solver=cp.OSQP,
                    verbose=False,
                    max_iter=100,
                    eps_rel=tol,
                    eps_abs=tol,
                )
                end_time = time.time()

                total_time += end_time - start_time
                if prob.status == "optimal":
                    Y.append(y.value.flatten())
                    obj_val.append(prob.value)
                else:
                    Y.append(np.ones(self.y_dim) * np.nan)
                    obj_val.append(np.nan)
            sols = np.array(Y)
            obj_val = np.array(obj_val)
            parallel_opt_time = total_time / len(X_np)
            print(f"{parallel_opt_time=}")
        else:
            raise NotImplementedError

        return sols, obj_val, total_time, parallel_opt_time

    def computeY(self):
        Y, obj_val, *_ = self.optimizationSolve(self.X, solver_type="osqp", tol=1e-5)
        feas_mask = ~np.isnan(Y).all(axis=1)
        self._nsamples = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        self._obj_val = torch.tensor(obj_val[feas_mask])
        return Y

    def updateInteriorPoint(self, Y0):
        self._Y0 = Y0
        return True
