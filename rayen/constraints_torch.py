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


class ConvexConstraints:
    def __init__(
        self,
        do_preprocessing_linear=False,
        print_debug_info=False,
    ):
        self.lc = LinearConstraint()
        # self.qcs = QuadraticConvexConstraint()
        # self.socs = SocConstraint()
        # self.lmis = LmiConstraint()

        self.has_linear_eq_constraints = False
        self.has_linear_ineq_constraints = False
        self.has_linear_constraints = False

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

        utils.verify(
            (
                self.has_quadratic_constraints
                or self.has_linear_constraints
                or self.has_soc_constraints
                or self.has_lmi_constraints
            ),
            "There are no constraints!",
        )
