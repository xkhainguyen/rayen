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
        self._P = None
        self._q = None
        self._A1 = None
        self._b1 = None
        self._Y = None
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
    def Y(self):
        return self._Y

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
    def trainX(self):
        return self.X[: self.train_num]

    @property
    def validX(self):
        return self.X[self.train_num : self.train_num + self.valid_num]

    @property
    def testX(self):
        return self.X[self.train_num + self.valid_num :]

    @property
    def trainY(self):
        return self.Y[: int(self.nsamples * self.train_frac)]

    @property
    def validY(self):
        return self.Y[
            int(self.nsamples * self.train_frac) : int(
                self.nsamples * (self.train_frac + self.valid_frac)
            )
        ]

    @property
    def testY(self):
        return self.Y[int(self.nsamples * (self.train_frac + self.valid_frac)) :]

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
        # shape (1, 1)
        return 0.5 * Y.transpose(-1, -2) @ self.P @ Y + self.q.transpose(-1, -2) @ Y

    def updateConstraints(self, Xc=None):
        if Xc is None:
            Xc_ = self.Xc
        else:
            Xc_ = Xc
        self._A1, self._b1, *_ = torch.vmap(self.cstrInputMap)(Xc_)
        return self._A1, self._b1

    def optimizationSolve(self, X, solver_type="osqp", tol=1e-4):
        if solver_type == "osqp":
            print("running osqp")
            P, q, A1, b1 = (
                self.P_np,
                self.q_np,
                self.A1_np,
                self.b1_np,
            )
            X_np = X.detach().cpu().numpy()
            Y = []
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
                    eps_prim_inf=tol,
                    max_iter=100,
                )
                start_time = time.time()
                results = solver.solve()
                end_time = time.time()

                total_time += end_time - start_time
                if results.info.status == "solved":
                    Y.append(results.x)
                else:
                    Y.append(np.ones(self.y_dim) * np.nan)

            # print(Y)
            sols = np.array(Y)
            parallel_time = total_time / len(X_np)

        else:
            raise NotImplementedError

        return sols, total_time, parallel_time

    def calc_Y(self):
        Y = self.optimizationSolve(self.X, tol=1e-8)[0]
        feas_mask = ~np.isnan(Y).all(axis=1)
        self._nsamples = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        return Y


###################################################################
# TEST
###################################################################
class CbfQpProblem(QpProblem):
    def __init__(self, X, xo_dim, xc_dim, y_dim, valid_frac=0.0833, test_frac=0.0833):
        super().__init__(X, xo_dim, xc_dim, y_dim, valid_frac, test_frac)

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
        b = torch.ones(dim * 2, 1) * 0.5  # CHANGE
        return A, b

    def getCbfConstraint(self, xc):
        # xc size (6, 1) including 3D pos and vel
        dim = self.y_dim
        alpha1 = 1.0
        alpha2 = 1.0
        pos_box = torch.ones(2 * dim, 1) * 1.2  # CHANGE
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


def test():
    num_samples = 2
    xo_dim = 1  # nominal control u_bar dimension
    xc_dim = 2  # state x dimension
    y_dim = 1
    # Xo = np.random.normal(0, 1, size=(num_samples, xo_dim))
    Xo = np.array([[1.0], [-1.0]])
    print(f"Xo = {Xo}")
    # Xc = np.random.normal(0, 1, size=(num_samples, xc_dim))
    Xc = np.array([[1.0, 0.3], [-1.0, -0.2]])
    print(f"Xc = {Xc}")
    X = np.hstack((Xo, Xc))
    print(f"X = {X}")
    problem = CbfQpProblem(
        X,
        xo_dim,
        xc_dim,
        y_dim,
        valid_frac=0.1,
        test_frac=0.1,
    )
    # print(f"{problem.X=}")
    # print(f"{problem.Xo=}")
    # print(f"{problem.Xc=}")
    # print(f"{problem.X_np=}")
    # print(f"{problem.Xo_np=}")
    # print(f"{problem.Xc_np=}")
    # print(f"{problem.trainX=}")
    # print(f"{problem.validX=}")
    # print(f"{problem.testX=}")
    # print(f"{problem.nsamples=} {problem.x_dim=} {problem.xo_dim=} {problem.xc_dim=}")

    problem.updateObjective()
    # print(f"{problem.P=}")
    # print(f"{problem.P_np=}")
    # print(f"{problem.q=}")
    # print(f"{problem.q_np=}")

    loss = problem.objectiveFunction(problem.Xo)
    # print(f"{loss=}")

    problem.updateConstraints()
    # print(f"{problem.A1=}")
    print(f"{problem.b1=}")
    # print(f"{problem.A1_np=}")

    problem.calc_Y()
    print(f"{problem.Y=}")


if __name__ == "__main__":
    test()
