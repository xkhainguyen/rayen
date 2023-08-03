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


class LinearConstraint:
    # Constraint is A1<=b1, A2=b2
    def __init__(
        self,
        A1=torch.tensor([]),
        b1=torch.tensor([]),
        A2=torch.tensor([]),
        b2=torch.tensor([]),
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
        return [A_p @ z - b_p <= -epsilon * torch.ones((A_p.shape[0], 1))]


class ConvexQuadraticConstraint:
    # Constraint is (1/2)x'Px + q'x +r <=0
    def __init__(
        self,
        P=torch.tensor([]),
        q=torch.tensor([]),
        r=torch.tensor([]),
        do_checks_P=False,
    ):
        self.P = P
        self.q = q
        self.r = r
        self.dim = 0

        if do_checks_P == True:
            utils.checkNonZeroTensor(self.P)
            utils.checkSymmetricTensor(self.P)

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
        if self.P.nelement() > 0:
            self.dim = self.P.shape[2]
        return self.dim

    def asCvxpy(self, y, P_sqrt, q, r, epsilon=0.0):
        return [
            0.5 * cp.sum_squares(P_sqrt @ y) + q.T @ y + r <= -epsilon
        ]  # assume_PSD needs to be True because of this: https://github.com/cvxpy/cvxpy/issues/407. We have already checked that it is Psd within a tolerance


class ConvexConstraints:
    def __init__(
        self,
        do_preprocessing_linear=False,
        print_debug_info=False,
    ):
        self.lc = LinearConstraint()
        self.qcs = ConvexQuadraticConstraint()
        # self.socs = SocConstraint()
        # self.lmis = LmiConstraint()

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

        self.has_quadratic_constraints = self.qcs.dim > 0
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

        utils.verify(utils.all_equal(all_dim))
