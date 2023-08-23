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


class SocProblem(ABC):
    """
    minimize_y  1/2 * y^T Po y + qo^T y
    s.t.        A2y =  b2
                A1y <= b1
                1/2 * y^T P y + q^T y + r <= 0
                ||M y + s|| - c^T y - d <= 0
    Po, qo come from objectiveFunction
    A1, b1, A2, b2, P, q, r, M, s, c, d come from constraintInpuMap
    """

    def __init__(
        self,
        X,
        xo_dim,
        xc_dim,
        y_dim,
        valid_frac=0.0833,
        test_frac=0.0833,
    ):
        self._X = torch.tensor(X).unsqueeze(-1)
        self._Po = None
        self._qo = None
        self._A1 = None
        self._b1 = None
        self._P = None
        self._P_sqrt = None
        self._q = None
        self._r = None
        self._M = None
        self._s = None
        self._c = None
        self._d = None
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
        return "SocProblem-{}-{}-{}".format(
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
    def Po(self):
        return self._Po

    @property
    def qo(self):
        return self._qo

    @property
    def A1(self):
        return self._A1

    @property
    def b1(self):
        return self._b1

    @property
    def P(self):
        return self._P

    @property
    def P_sqrt(self):
        return self._P_sqrt

    @property
    def q(self):
        return self._q

    @property
    def r(self):
        return self._r

    @property
    def M(self):
        return self._M

    @property
    def s(self):
        return self._s

    @property
    def c(self):
        return self._c

    @property
    def d(self):
        return self._d

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
    def Y(self):
        return self._Y

    @property
    def obj_val(self):
        return self._obj_val

    @property
    def Po_np(self):
        return self.Po.detach().cpu().numpy()

    @property
    def qo_np(self):
        return self.qo.detach().cpu().numpy()

    @property
    def A1_np(self):
        return self.A1.detach().cpu().numpy()

    @property
    def b1_np(self):
        return self.b1.detach().cpu().numpy()

    @property
    def P_np(self):
        return self.P.detach().cpu().numpy()

    @property
    def P_sqrt_np(self):
        return self.P.detach().cpu().numpy()

    @property
    def q_np(self):
        return self.q.detach().cpu().numpy()

    @property
    def r_np(self):
        return self.r.detach().cpu().numpy()

    @property
    def M_np(self):
        return self.M.detach().cpu().numpy()

    @property
    def s_np(self):
        return self.s.detach().cpu().numpy()

    @property
    def c_np(self):
        return self.c.detach().cpu().numpy()

    @property
    def d_np(self):
        return self.d.detach().cpu().numpy()

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
        self._Po, self._qo = torch.vmap(self.objInputMap)(Xo_)
        return self._Po, self._qo

    ## For PyTorch
    def objectiveFunction(self, Y):
        # shape (1, 1)
        return 0.5 * Y.transpose(-1, -2) @ self.Po @ Y + self.qo.transpose(-1, -2) @ Y

    def updateConstraints(self, Xc=None):
        if Xc is None:
            Xc_ = self.Xc
        else:
            Xc_ = Xc
        (
            self._A1,
            self._b1,
            _,
            _,
            self._P,
            self._P_sqrt,
            self._q,
            self._r,
            self._M,
            self._s,
            self._c,
            self._d,
            *_,
        ) = torch.vmap(self.cstrInputMap)(Xc_)
        return (
            self._A1,
            self._b1,
            self._P,
            self._P_sqrt,
            self._q,
            self._r,
            self._M,
            self._s,
            self._c,
            self._d,
        )

    def optimizationSolve(self, X, solver_type="cvxpy_ecos", tol=1e-5):
        if solver_type == "cvxpy_ecos":
            utils.printInBoldBlue("running cvxpy with ecos")
            Po, qo, A1, b1, P, q, r, M, s, c, d = (
                self.Po_np,
                self.qo_np,
                self.A1_np,
                self.b1_np,
                self.P_np,
                self.q_np,
                self.r_np,
                self.M_np,
                self.s_np,
                self.c_np,
                self.d_np,
            )
            X_np = X.detach().cpu().numpy()
            Y = []
            obj_val = []
            total_time = 0
            for i in range(self.nsamples):
                y = cp.Variable((self.y_dim, 1))
                # print(P, q, r, A1, b1)
                cs = [A1[i] @ y <= b1[i]]
                cs += [(1 / 2) * cp.quad_form(y, P[i]) + q[i].T @ y + r[i] <= 0.0]
                cs += [
                    cp.norm(self.M[i] @ y + self.s[i]) - self.c[i].T @ y - self.d[i]
                    <= 0.0
                ]
                prob = cp.Problem(
                    cp.Minimize((1 / 2) * cp.quad_form(y, Po[i]) + qo[i].T @ y), cs
                )

                start_time = time.time()
                prob.solve(
                    solver=cp.ECOS,
                    verbose=False,
                    max_iters=100,
                    abstol=tol,
                    reltol=tol,
                    feastol=tol,
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

    def computeY(self, tol=1e-8):
        Y, obj_val, *_ = self.optimizationSolve(
            self.X, solver_type="cvxpy_ecos", tol=tol
        )
        feas_mask = ~np.isnan(Y).all(axis=1)
        self._nsamples = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        self._obj_val = torch.tensor(obj_val[feas_mask])
        return Y

    @property
    def Y0_np(self):
        return self.Y0.detach().cpu().numpy()

    @property
    def Y0(self):
        return self._Y0

    def updateInteriorPoint(self, Y0):
        self._Y0 = Y0
        return True
