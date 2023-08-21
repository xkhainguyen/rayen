# --------------------------------------------------------------------------
# Jesus Tordesillas Torres, Robotic Systems Lab, ETH Zürich
# See LICENSE file for the license information
# --------------------------------------------------------------------------

import torch
import torch.nn as nn
from . import utils
import numpy as np
import cvxpy as cp
import math
from cvxpylayers.torch import CvxpyLayer
import random
import copy
import time


################### CONSTRAINTS
# everything is torch


class LinearConstraints:
    # Constraint is A1<=b1, A2=b2
    # A and b are tensors of batched samples
    def __init__(
        self,
        A1=torch.tensor([]),
        b1=torch.tensor([]),
        A2=torch.tensor([]),
        b2=torch.tensor([]),
        num=None,
    ):
        self.A1 = A1
        self.b1 = b1
        self.A2 = A2
        self.b2 = b2
        self.dim = 0

        # utils.verify(self.hasEqConstraints() or self.hasIneqConstraints())

        # if self.hasIneqConstraints():
        #     utils.verify(A1.ndim == 2)
        #     utils.verify(b1.ndim == 2)
        #     utils.verify(b1.shape[1] == 1)
        #     utils.verify(A1.shape[0] == b1.shape[0])

        # if self.hasEqConstraints():
        #     utils.verify(A2.ndim == 2)
        #     utils.verify(b2.ndim == 2)
        #     utils.verify(b2.shape[1] == 1)
        #     utils.verify(A2.shape[0] == b2.shape[0])

        # if self.hasIneqConstraints() and self.hasEqConstraints():
        #     utils.verify(A1.shape[1] == A2.shape[1])

    def hasEqConstraints(self):
        return self.A2.nelement() > 0 and self.b2.nelement() > 0

    def hasIneqConstraints(self):
        return self.A1.nelement() > 0 and self.b1.nelement() > 0

    def getDim(self):
        if self.hasIneqConstraints():
            self.dim = self.A1.shape[2]
        if self.hasEqConstraints():
            self.dim = self.A2.shape[2]
        return 0

    def asCvxpy(self, y, epsilon=0.0):
        constraints = []
        if self.hasIneqConstraints():
            constraints.append(self.A1 @ y <= self.b1)
        if self.hasEqConstraints():
            constraints.append(self.A2 @ y == self.b2)
        return constraints

    def asCvxpySubspace(self, z, A_p, b_p, epsilon=0.0):
        # print(torch.ones((A_p.shape[0], 1), device="cpu"))
        print(A_p)
        return [A_p @ z - b_p <= b_p]


class ConvexQuadraticConstraints:
    # Constraint in form (1/2)x'P[i]x + q[i]'x +r[i] <=0
    # Include all P, q, r stacked in a sample
    def __init__(
        self,
        P=torch.tensor([]),
        P_sqrt=torch.tensor([]),
        q=torch.tensor([]),
        r=torch.tensor([]),
        num=0,
        do_checks_P=False,
    ):
        self.P = P
        self.P_sqrt = P_sqrt
        self.q = q
        self.r = r
        self.dim = 0
        self.num = num

        # if do_checks_P:
        #     utils.checkNonZeroTensor(self.P)
        #     utils.checkSymmetricTensor(self.P)

        # eigenvalues = np.linalg.eigvalsh(self.P)
        # smallest_eigenvalue = np.amin(eigenvalues)

        ######## Check that the matrix is PSD up to a tolerance
        # tol = 1e-7
        # utils.verify(
        #     smallest_eigenvalue > -tol,
        #     f"Matrix P is not PSD, smallest eigenvalue is {smallest_eigenvalue}",
        # )
        #########################

        # Note: All the code assummes that P is a PSD matrix. This is specially important when:
        # --> Using  cp.quad_form(...) You can use the argument assume_PSD=True (see https://github.com/cvxpy/cvxpy/issues/407)
        # --> Computting kappa (if P is not a PSD matrix, you end up with a negative discriminant when solving the 2nd order equation)

        ######### Correct for possible numerical errors
        # if (-tol) <= smallest_eigenvalue < 0:
        #     # Correction due to numerical errors

        #     ##Option 1
        #     self.P = self.P + np.abs(smallest_eigenvalue) * np.eye(self.P.shape[0])

        ##Option 2 https://stackoverflow.com/a/63131250  and https://math.stackexchange.com/a/1380345
        # C = (self.P + self.P.T)/2  #https://en.wikipedia.org/wiki/Symmetric_matrix#Decomposition_into_symmetric_and_skew-symmetric
        # eigval, eigvec = np.linalg.eigh(C)
        # eigval[eigval < 0] = 0
        # self.P=eigvec.dot(np.diag(eigval)).dot(eigvec.T)
        ##########

    def getDim(self):
        # Within batch
        if self.P.nelement() > 0:
            self.dim = self.P.shape[2]
        return self.dim

    def at(self, i):
        # Within sample
        return range(i * self.dim, (i + 1) * self.dim)

    def asCvxpy(self, y, P_sqrt, q, r, epsilon=0.0):
        # Within sample, one constraint

        # assume_PSD needs to be True because of this: https://github.com/cvxpy/cvxpy/issues/407. We have already checked that it is Psd within a tolerance
        return [0.5 * cp.sum_squares(P_sqrt @ y) + q.T @ y + r <= -epsilon]


class SocConstraint:
    # Constraint is ||My+s||<=c'y+d
    def __init__(
        self,
        M=torch.tensor([]),
        s=torch.tensor([]),
        c=torch.tensor([]),
        d=torch.tensor([]),
        num=0,
    ):
        # utils.checkMatrixisNotZero(M)
        # utils.checkMatrixisNotZero(c)

        # utils.verify(M.shape[1] == c.shape[0])
        # utils.verify(M.shape[0] == s.shape[0])
        # utils.verify(s.shape[1] == 1)
        # utils.verify(c.shape[1] == 1)
        # utils.verify(d.shape[0] == 1)
        # utils.verify(d.shape[1] == 1)

        self.M = M
        self.s = s
        self.c = c
        self.d = d
        self.num = num
        self.dim = 0

    def getDim(self):
        # Within batch
        if self.M.nelement() > 0:
            self.dim = self.M.shape[2]
        return self.dim

    def at(self, i):
        # Within sample
        return range(i * self.dim, (i + 1) * self.dim)

    def asCvxpy(self, y, M, s, c, d, epsilon=0.0):
        return [cp.norm(M @ y + s) - c.T @ y - d <= -epsilon]


class LmiConstraint:
    # Constraint is y0 F0 + y1 F1 + ... + ykm1 Fkm1 + Fk >=0
    # Stack vertically F = [F0; F1; F2;...;Fk]
    def __init__(self, F=torch.tensor([])):
        self.F = F
        self.dim = 0
        self.Fdim = 0

    def getDim(self):
        # Within batch
        if self.F.nelement() > 0:
            self.Fdim = self.F.shape[2]
            self.dim = int(self.F.shape[1] / self.Fdim) - 1
        return self.dim

    def at(self, i):
        # Within sample
        return range(i * self.Fdim, (i + 1) * self.Fdim)

    def asCvxpy(self, y, F, epsilon=0.0):
        lmi_left_hand_side = 0
        k = self.dim
        for i in range(k):
            idx = self.at(i)
            lmi_left_hand_side += y[i, 0] * F[idx, :]
        lmi_left_hand_side += F[self.at(k), :]

        return [lmi_left_hand_side >> epsilon * torch.eye(self.Fdim)]


class ConvexConstraints:
    def __init__(
        self,
        num_cstr=None,
        do_preprocessing_linear=False,
        print_debug_info=False,
    ):
        self.num_lineq = num_cstr[0]  # number of linear inequalities
        self.num_leq = num_cstr[1]  # number of linear equalities
        self.num_qc = num_cstr[2]  # number of convex quadratic constraints
        self.num_soc = num_cstr[3]  # number of SOC constraints
        self.num_lmi = num_cstr[4]  # number of LMI constraints

        self.lc = LinearConstraints()
        self.qcs = ConvexQuadraticConstraints(num=self.num_qc)
        self.socs = SocConstraint(num=self.num_soc)
        self.lmis = LmiConstraint()

        self.has_linear_eq_constraints = False
        self.has_linear_ineq_constraints = False
        self.has_linear_constraints = False

        self.has_nonlinear_constraints = False
        self.has_quadratic_constraints = False
        self.has_soc_constraints = False
        self.has_lmi_constraints = False

    def firstInit(self):
        self.has_linear_eq_constraints = self.lc.hasEqConstraints()
        self.has_linear_ineq_constraints = self.lc.hasIneqConstraints()
        self.has_linear_constraints = (
            self.has_linear_eq_constraints or self.has_linear_ineq_constraints
        )
        self.lc.getDim()
        self.qcs.getDim()
        self.socs.getDim()
        self.lmis.getDim()

        self.has_quadratic_constraints = self.qcs.dim > 0
        self.has_soc_constraints = self.socs.dim > 0
        self.has_lmi_constraints = self.lmis.dim > 0
        self.has_nonlinear_constraints = (
            self.has_quadratic_constraints
            or self.has_soc_constraints
            or self.has_lmi_constraints
        )

        utils.verify(
            (
                self.has_quadratic_constraints
                or self.has_linear_constraints
                or self.has_soc_constraints
                or self.has_lmi_constraints
            ),
            "There are no constraints!",
        )

        # Check that the dimensions of all the constraints are the same
        all_dim = []
        if self.has_linear_constraints:
            all_dim.append(self.lc.dim)
        if self.has_quadratic_constraints:
            all_dim.append(self.qcs.dim)
        if self.has_soc_constraints:
            all_dim.append(self.socs.dim)
        if self.has_lmi_constraints:
            all_dim.append(self.lmis.dim)

        utils.verify(utils.all_equal(all_dim), "wrong constraint dimension")

        if self.has_quadratic_constraints:
            utils.verify(
                self.qcs.P.shape[1] == self.qcs.dim * self.qcs.num
                and self.qcs.P.shape[2] == self.qcs.dim,
                "wrong quadratic constraint P dimension",
            )

            utils.verify(
                self.qcs.P_sqrt.shape[1] == self.qcs.dim * self.qcs.num
                and self.qcs.P_sqrt.shape[2] == self.qcs.dim,
                "wrong quadratic constraint P_sqrt dimension",
            )

            utils.verify(
                self.qcs.q.shape[1] == self.qcs.dim * self.qcs.num
                and self.qcs.q.shape[2] == 1,
                "wrong quadratic constraint q dimension",
            )

            utils.verify(
                self.qcs.r.shape[1] == self.qcs.num and self.qcs.r.shape[2] == 1,
                "wrong quadratic constraint r dimension",
            )

        if self.has_soc_constraints:
            utils.verify(
                self.socs.M.shape[1] == self.socs.dim * self.socs.num
                and self.socs.M.shape[2] == self.socs.dim,
                "wrong SOC constraint M dimension",
            )

            utils.verify(
                self.socs.s.shape[1] == self.socs.dim * self.socs.num
                and self.socs.s.shape[2] == 1,
                "wrong SOC constraint s dimension",
            )

            utils.verify(
                self.socs.c.shape[1] == self.socs.dim * self.socs.num
                and self.socs.c.shape[2] == 1,
                "wrong SOC constraint c dimension",
            )

            utils.verify(
                self.socs.d.shape[1] == self.socs.num and self.socs.d.shape[2] == 1,
                "wrong SOC constraint d dimension",
            )
