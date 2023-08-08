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
from . import constraints_torch


class ConstraintModule(torch.nn.Module):
    """Description:
    Args:
     xv_dim: dimension of step input sample
     xc_dim: dimension of constraint input sample
     y_dim: dimension of output, ambient space
     method: method to use
     num_cstr: number of each type of constraints
     args_DC3: arguments for DC3
    """

    def __init__(
        self,
        xv_dim=None,
        xc_dim=None,
        y_dim=None,
        method="RAYEN",
        num_cstr=None,
        constraintInputMap=None,
        args_DC3=None,
    ):
        super().__init__()

        self.method = method
        self.m = xv_dim  # Dimension of the step input vector xv
        self.d = xc_dim  # Dimension of the constraint input vector xc
        self.k = y_dim  # Dimension of the ambient space (output)
        self.n = None  # Dimension of the embedded space (determined later)
        self.batch_size = None
        self.constraintInputMap = constraintInputMap

        self.cs = constraints_torch.ConvexConstraints(num_cstr)
        self.params_indexes = {"qc": None, "soc": None, "lmi": None}

        # Pass dummy input to check the constraintInputMap users provide
        self.test_dummy(xc_dim)

        self.cs.firstInit()

        # Get the dimension of all potential linear constraints stacked together
        if self.cs.has_linear_constraints:
            self.Ap_nrows = self.cs.lc.A1.shape[1]
        else:
            self.Ap_nrows = 1

        self.selectSolver()

        if self.method == "RAYEN":
            # Handle dimensions of ambient and embedded space
            if self.cs.has_linear_eq_constraints:
                self.n = self.k - 1  # Just one linear equality constraint
            else:
                self.n = self.k

            create_step_input_map = True if self.n != self.m else False

            self.setupInteriorPointLayer()

            self.forwardForMethod = self.forwardForRAYEN

        if create_step_input_map:
            self.stepInputMap = nn.Linear(self.m, self.n)
        else:
            self.stepInputMap = nn.Sequential()

    def test_dummy(self, xc_dim):
        temp_x = torch.ones(1, xc_dim, 1)  # just a vector
        (
            self.cs.lc.A1,
            self.cs.lc.b1,
            self.cs.lc.A2,
            self.cs.lc.b2,
            self.cs.qcs.P,
            self.cs.qcs.P_sqrt,
            self.cs.qcs.q,
            self.cs.qcs.r,
            self.cs.socs.M,
            self.cs.socs.s,
            self.cs.socs.c,
            self.cs.socs.d,
            self.cs.lmis.F,
        ) = torch.vmap(self.constraintInputMap)(temp_x)

    def selectSolver(self):
        installed_solvers = cp.installed_solvers()
        if ("GUROBI" in installed_solvers) and self.cs.has_lmi_constraints == False:
            self.solver = "GUROBI"  # You need to do `python -m pip install gurobipy`
        if ("ECOS" in installed_solvers) and self.cs.has_lmi_constraints == False:
            self.solver = "ECOS"
        elif "SCS" in installed_solvers:
            self.solver = "SCS"
        # elif 'OSQP' in installed_solvers:
        # 	self.solver='OSQP'
        # elif 'CVXOPT' in installed_solvers:
        # 	self.solver='CVXOPT'
        else:
            # There are more solvers, see https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver
            raise Exception(f"Which solver do you have installed?")
        # self.solver = "ECOS"
        return True

    # Function to recompute and register params
    def updateForwardParams(self):
        self.y0 = self.gety0()
        # Precompute for inverse distance to the frontier of Z along v_bar
        self.D = self.A_p / (
            (self.b_p - self.A_p @ self.z0) @ torch.ones(self.batch_size, 1, self.n)
        )  # for linear

        if self.cs.has_quadratic_constraints:
            self.all_delta = torch.empty((self.batch_size, 0, self.n))
            self.all_phi = torch.empty((self.batch_size, 0, self.n))
            for i in range(self.cs.num_qc):
                idx = self.cs.qcs.at(i)
                P = self.cs.qcs.P[:, idx, :]
                q = self.cs.qcs.q[:, idx, :]
                r = self.cs.qcs.r[:, i, :].unsqueeze(1)
                y0 = self.y0
                y0t = y0.transpose(-1, -2)
                qt = q.transpose(-1, -2)

                sigma = 2 * (0.5 * y0t @ P @ y0 + qt @ y0 + r)
                phi = -(y0t @ P + qt) / sigma
                delta = (
                    (y0t @ P + qt).transpose(-1, -2) @ (y0t @ P + qt)
                    - 4 * (0.5 * y0t @ P @ y0 + qt @ y0 + r) * 0.5 * P
                ) / torch.square(sigma)
                self.all_phi = torch.cat((self.all_phi, phi), dim=1)
                self.all_delta = torch.cat((self.all_delta, delta), dim=1)

        if self.cs.has_lmi_constraints:
            idx = self.cs.lmis.at(self.k)
            H = self.cs.lmis.F[:, idx, :]
            for i in range(self.k):
                idx = self.cs.lmis.at(i)
                H += self.y0[:, i, :].unsqueeze(1) * self.cs.lmis.F[:, idx, :]
            Hinv = torch.linalg.inv(H)
            # self.mHinv = -Hinv
            # print(self.mHinv)
            self.L = torch.linalg.cholesky(Hinv)  # Hinv = L @ L^T
            # print(self.L)
        return True

    # From constraint input batch X, update all constraint data batch
    def updateSubspaceConstraints(self):
        if self.cs.has_linear_constraints:
            # Retrive data
            A1 = self.cs.lc.A1
            b1 = self.cs.lc.b1
            A2 = self.cs.lc.A2
            b2 = self.cs.lc.b2

            # Stack the matrices so that the linear constraints look like Ax<=b
            if self.cs.has_linear_ineq_constraints:
                A = A1
                b = b1
                if self.cs.has_linear_eq_constraints:
                    # Add the equality constraints as inequality constraints
                    A = torch.cat((A, A2, -A2), axis=1)
                    b = torch.cat((b, b2, -b2), axis=1)
            else:
                # Add the equality constraints as inequality constraints
                A = torch.cat((A, A2, -A2), axis=1)
                b = torch.cat((b, b2, -b2), axis=1)

            print_debug_info = 1
            if print_debug_info:
                utils.printInBoldGreen(f"A is {A.shape} and b is {b.shape}")

            # TODO Preprocess
            # Here we simply choose E such that
            # A_E == A2, b_E == b_2
            # A_I == A1, b_I == b_1
            if self.cs.has_linear_ineq_constraints:
                start = A1.shape[1]
            else:
                start = 0
            E = list(range(start, A.shape[1]))

            if print_debug_info:
                utils.printInBoldGreen(f"E={E}")

            I = [i for i in range(A.shape[1]) if i not in E]

            # Obtain A_E, b_E and A_I, b_I
            if len(E) > 0:
                A_E = A[:, E, :]
                b_E = b[:, E, :]
            else:
                A_E = torch.zeros(self.batch_size, 1, A.shape[2])
                b_E = torch.zeros(self.batch_size, 1, 1)

            if len(I) > 0:
                A_I = A[:, I, :]
                b_I = b[:, I, :]
            else:
                A_I = torch.zeros(self.batch_size, 1, A.shape[2])
                # 0z<=1
                b_I = torch.ones(self.batch_size, 1, 1)

            if print_debug_info:
                utils.printInBoldGreen(f"AE={A_E}")
                utils.printInBoldGreen(f"AI={A_I}")

            # At this point, A_E, b_E, A_I, and b_I have at least one row

            # Project into the nullspace of A_E
            ################################################
            NA_E = nullSpace(A_E)
            # print(f"NA_E = {NA_E}")
            # n = NA_E.shape[2]  # dimension of the subspace

            # Only take the half of A_E, b_E
            yp = torch.linalg.lstsq(A_E[:, 0:1, :], b_E[:, 0:1, :]).solution
            # print(f"yp = {yp}")
            A_p = A_I @ NA_E
            b_p = b_I - A_I @ yp

            utils.verify(A_p.ndim == 3, f"A_p.shape={A_p.shape}")
            utils.verify(b_p.ndim == 3, f"b_p.shape={b_p.shape}")
            utils.verify(b_p.shape[2] == 1)
            utils.verify(A_p.shape[1] == b_p.shape[1])

            # if print_debug_info:
            #     utils.printInBoldGreen(f"A_p is {A_p.shape} and b_p is {b_p.shape}")

            self.n = A_p.shape[2]  # dimension of the linear subspace
            # print(self.n)
        else:
            self.n = self.k
            NA_E = torch.eye(self.n).unsqueeze(0).repeat(self.batch_size, 1, 1)
            yp = torch.zeros(self.batch_size, self.k, 1)
            # 0z<=1
            A_p = torch.zeros(self.batch_size, 1, self.n)
            b_p = torch.ones(self.batch_size, 1, 1)
            A_E = torch.zeros(self.batch_size, 1, self.n)
            # 0y=0
            b_E = torch.zeros(self.batch_size, 1, 1)
            A_I = torch.zeros(self.batch_size, 1, self.n)
            # 0y<=1
            b_I = torch.ones(self.batch_size, 1, 1)

        self.A_E = A_E
        self.b_E = b_E
        self.A_I = A_I
        self.b_I = b_I

        self.A_p = A_p
        self.b_p = b_p
        self.yp = yp
        self.NA_E = NA_E

        # print(f"A_E = {self.A_E}")
        # print(f"b_E = {self.b_E}")
        # print(f"A_I = {self.A_I}")
        # print(f"A_p = {self.A_p}")
        # print(f"b_p = {self.b_p}")
        # print(f"NA_E = {self.NA_E}")
        # print(f"yp = {self.yp}")
        # print(f"z0 = {self.cs.z0}")

        # utils.verify(self.n == (self.k - np.linalg.matrix_rank(self.A_E[0])))
        return True

    def getConstraintsInSubspaceCvxpy(self, z, y, params=None, epsilon=0.0):
        NA_E, yp, A_p, b_p, *_ = params
        constraints = self.cs.lc.asCvxpySubspace(z, A_p, b_p, epsilon)

        constraints += [y == NA_E @ z + yp]

        if self.cs.has_quadratic_constraints:
            cs_idx = self.params_indexes["qc"]
            P_sqrt, q, r = params[cs_idx : cs_idx + 3]
            for i in range(self.cs.num_qc):
                idx = self.cs.qcs.at(i)
                constraints += self.cs.qcs.asCvxpy(
                    y, P_sqrt[idx, :], q[idx, :], r[i], epsilon
                )

        if self.cs.has_soc_constraints:
            cs_idx = self.params_indexes["soc"]
            M, s, c, d = params[cs_idx : cs_idx + 4]
            for i in range(self.cs.num_soc):
                idx = self.cs.socs.at(i)
                constraints += self.cs.socs.asCvxpy(
                    y, M[idx, :], s[idx, :], c[idx, :], d[i], epsilon
                )

        if self.cs.has_lmi_constraints:
            cs_idx = self.params_indexes["lmi"]
            F = params[cs_idx]
            constraints += self.cs.lmis.asCvxpy(y, F, epsilon)

        return constraints

    # Setup interior point problem beforehand
    def setupInteriorPointLayer(self):
        self.ip_epsilon = cp.Variable()
        self.ip_z0 = cp.Variable((self.n, 1))
        self.ip_y = cp.Variable((self.k, 1))

        NA_E = cp.Parameter((self.k, self.n))
        yp = cp.Parameter((self.k, 1))

        A_p = cp.Parameter((self.Ap_nrows, self.n))
        b_p = cp.Parameter((self.Ap_nrows, 1))

        params = [NA_E, yp, A_p, b_p]

        if self.cs.has_quadratic_constraints:
            P_sqrt = cp.Parameter((self.n * self.cs.num_qc, self.n))
            q = cp.Parameter((self.n * self.cs.num_qc, 1))
            r = cp.Parameter((self.cs.num_qc, 1))
            self.params_indexes["qc"] = len(params)
            params += [P_sqrt, q, r]

        if self.cs.has_soc_constraints:
            M = cp.Parameter((self.n * self.cs.num_soc, self.n))
            s = cp.Parameter((self.n * self.cs.num_soc, 1))
            c = cp.Parameter((self.n * self.cs.num_soc, 1))
            d = cp.Parameter((self.cs.num_soc, 1))
            self.params_indexes["soc"] = len(params)
            params += [M, s, c, d]

        if self.cs.has_lmi_constraints:
            F = cp.Parameter((self.cs.lmis.F.shape[1], self.cs.lmis.Fdim))
            self.params_indexes["lmi"] = len(params)
            params += [F]

        self.ip_constraints = self.getConstraintsInSubspaceCvxpy(
            self.ip_z0,
            self.ip_y,
            params=params,
            epsilon=self.ip_epsilon,
        )

        # print(self.ip_constraints)
        self.ip_constraints.append(self.ip_epsilon >= 0)
        self.ip_constraints.append(
            self.ip_epsilon <= 0.5
        )  # This constraint is needed for the case where the set is unbounded. Any positive value is valid

        objective = cp.Minimize(-self.ip_epsilon)
        self.ip_prob = cp.Problem(objective, self.ip_constraints)

        # assert self.ip_prob.is_dpp()
        self.ip_layer = CvxpyLayer(
            self.ip_prob,
            parameters=params,
            variables=[self.ip_z0, self.ip_epsilon, self.ip_y],
        )

        print_debug_info = 0
        if print_debug_info:
            utils.printInBoldBlue("Set up interior point problem")
        return True

    # Function to compute z0
    def solveInteriorPoint(self):
        # Update constraint
        # self.ip_constraints = self.cs.getConstraintsInSubspaceCvxpy(
        #     self.ip_z0, self.ip_epsilon
        # )
        # print(f"self.A_p = {self.A_p}")
        # print(f"self.b_p = {self.b_p}")
        # print(f"P = {self.cs.qcs.P}")
        # print(f"q = {self.cs.qcs.q}")
        # print(f"r = {self.cs.qcs.r}")
        params = [self.NA_E, self.yp, self.A_p, self.b_p]
        if self.cs.has_quadratic_constraints:
            params += [self.cs.qcs.P_sqrt, self.cs.qcs.q, self.cs.qcs.r]
        if self.cs.has_soc_constraints:
            params += [self.cs.socs.M, self.cs.socs.s, self.cs.socs.c, self.cs.socs.d]
        if self.cs.has_lmi_constraints:
            params += [self.cs.lmis.F]

        ip_z0, ip_epsilon, ip_y = self.ip_layer(
            *params, solver_args={"solve_method": self.solver}
        )  # "max_iters": 10000
        # print(f"epsilon = {ip_epsilon}")

        # if ip_prob.status != "optimal" and ip_prob.status != "optimal_inaccurate":
        #     raise Exception(f"Value is not optimal, prob_status={ip_prob.status}")

        # utils.verify(
        #     epsilon.value > 1e-8
        # )  # If not, there are no strictly feasible points in the subspace

        return ip_z0

    def computeKappa(self, v_bar):
        kappa = torch.relu(torch.max(self.D @ v_bar, dim=1, keepdim=True).values)

        if self.cs.has_nonlinear_constraints:
            rho = self.NA_E @ v_bar
            rhoT = torch.transpose(rho, -1, -2)
            all_kappas_positives = torch.empty((self.batch_size, 0, 1))

            # for each of the quadratic constraints
            for i in range(self.cs.num_qc):
                idx = self.cs.qcs.at(i)
                kappa_positive_i = self.all_phi[:, i, :] @ rho + torch.sqrt(
                    rhoT @ self.all_delta[:, idx, :] @ rho
                )
                assert torch.all(
                    kappa_positive_i >= 0
                ), f"Smallest element is {kappa_positive_i}"  # If not, then Z may not be feasible (note that z0 is in the interior of Z)
                all_kappas_positives = torch.cat(
                    (all_kappas_positives, kappa_positive_i), dim=1
                )

            # for each of the SOC constraints
            for i in range(self.cs.num_soc):
                idx = self.cs.socs.at(i)
                M = self.cs.socs.M[:, idx, :]
                s = self.cs.socs.s[:, idx, :]
                c = self.cs.socs.c[:, idx, :]
                d = self.cs.socs.d[:, i, :].unsqueeze(1)
                cT = c.transpose(-1, -2)

                beta = M @ self.y0 + s
                tau = cT @ self.y0 + d

                c_prime = rhoT @ M.transpose(-1, -2) @ M @ rho - torch.square(cT @ rho)
                b_prime = 2 * rhoT @ M.transpose(-1, -2) @ beta - 2 * (cT @ rho) @ tau
                a_prime = beta.transpose(-1, -2) @ beta - torch.square(tau)

                kappa_positive_i = self.solveSecondOrderEq(
                    a_prime, b_prime, c_prime, False
                )
                # print(kappa_positive_i)
                assert torch.all(
                    kappa_positive_i >= 0
                )  # If not, then either the feasible set is infeasible (note that z0 is inside the feasible set)
                all_kappas_positives = torch.cat(
                    (all_kappas_positives, kappa_positive_i), dim=1
                )
                # print(all_kappas_positives)

            if self.cs.has_lmi_constraints:  # If there are LMI constraints:
                # print("LMI")
                ############# OBTAIN S
                # First option (much slower)
                S = torch.zeros(self.batch_size, self.cs.lmis.Fdim, self.cs.lmis.Fdim)
                for i in range(self.cs.lmis.dim):
                    idx = self.cs.lmis.at(i)
                    S += self.cs.lmis.F[:, idx, :] * rho[:, i, :].unsqueeze(1)
                # Second option (much faster)
                # S = torch.einsum(
                #     "ajk,ial->ijk", [self.cs.lmis.F[0:-1, :, :], rho]
                # )  # See the tutorial https://rockt.github.io/2018/04/30/einsum
                # print(S)
                ############# COMPUTE THE EIGENVALUES

                ## Option 1: (compute whole spectrum of the matrix, using the non-symmetric matrix self.mHinv@S)
                # eigenvalues = torch.unsqueeze(torch.linalg.eigvals(self.mHinv@S),2) #Note that mHinv@M is not symmetric but always have real eigenvalues
                # assert (torch.all(torch.isreal(eigenvalues)))
                # largest_eigenvalue = torch.max(eigenvalues.real, dim=1, keepdim=True).values

                LTmSL = (
                    self.L.transpose(-1, -2) @ (-S) @ self.L
                )  # This matrix is symmetric

                ## Option 2: (compute whole spectrum of the matrix, using the symmetric matrix LTmSL). Much faster than Option 1
                eigenvalues = torch.unsqueeze(
                    torch.linalg.eigvalsh(LTmSL), 2
                )  # Note that L^T (-S) L is a symmetric matrix
                largest_eigenvalue = torch.max(eigenvalues, dim=1, keepdim=True).values

                ## Option 3: Use LOBPCG with A=LTmSL and B=I. The advantage of this method is that only the largest eigenvalue is computed. But, empirically, this option is faster than option 2 only for very big matrices (>1000x1000)
                # guess_lobpcg=torch.rand(1, H.shape[0], 1);
                # size_batch=v_bar.shape[0]
                # largest_eigenvalue, _ = torch.lobpcg(A=LTmSL, k=1, B=None, niter=-1) #, X=guess_lobpcg.expand(size_batch, -1, -1)
                # largest_eigenvalue=torch.unsqueeze(largest_eigenvalue, 1)

                ## Option 4: Use power iteration to compute the largest eigenvalue. Often times is slower than just computing the whole spectrum, and sometimes it does not converge
                # guess_v = torch.nn.functional.normalize(torch.rand(S.shape[1],1), dim=0)
                # largest_eigenvalue=utils.findLargestEigenvalueUsingPowerIteration(self.mHinv@S, guess_v)

                ## Option 5: Use LOBPCG with A=-S and B=H. There are two problems though:
                # --> This issue: https://github.com/pytorch/pytorch/issues/101075
                # --> Backward is not implemented for B!=I, see: https://github.com/pytorch/pytorch/blob/d54fcd571af48685b0699f6ac1e31b6871d0d768/torch/_lobpcg.py#L329

                ## Option 6: Use https://github.com/rfeinman/Torch-ARPACK with LTmSL. The problem is that backward() is not implemented yet

                ## Option 7: Use https://github.com/buwantaiji/DominantSparseEigenAD. But it does not have support for batched matrices, see https://github.com/buwantaiji/DominantSparseEigenAD/issues/1

                kappa_positive_i = torch.relu(largest_eigenvalue)

                all_kappas_positives = torch.cat(
                    (all_kappas_positives, kappa_positive_i), dim=1
                )

            kappa_nonlinear_constraints = torch.max(
                all_kappas_positives, dim=1, keepdim=True
            ).values
            kappa = torch.maximum(kappa, kappa_nonlinear_constraints)

        assert torch.all(kappa >= 0)
        # print(kappa)
        return kappa

    # Forward pass for RAYEN
    def forwardForRAYEN(self, v):
        # Update Ap, bp, NA_E
        self.updateSubspaceConstraints()  # torch!!

        # Solve interior point
        self.z0 = self.solveInteriorPoint()
        # print(f"z0 = {self.z0}")

        # # Update and register all necessary parameters
        self.updateForwardParams()

        v_bar = torch.nn.functional.normalize(v, dim=1)
        kappa = self.computeKappa(v_bar)
        norm_v = torch.linalg.vector_norm(v, dim=(1, 2), keepdim=True)
        # print(norm_v)
        alpha = torch.minimum(1 / kappa, norm_v)
        # print(f"alpha = {alpha}")
        return self.getyFromz(self.z0 + alpha * v_bar)

    def forward(self, x):
        ##################  MAPPER LAYER ####################
        # nn.Module forward method only accepts a single input tensor
        # nsib denotes the number of samples in the batch
        # each sample includes xv (size m) and xc (size d)
        # x has dimensions [nsib, m + d, 1]
        self.batch_size = x.shape[0]
        utils.verify(x.shape[1] == self.m + self.d, "wrong input dimension")
        xv = x[:, 0 : self.m, 0:1]  # After this, xv has dim [nsib, m, 1]
        xc = x[:, self.m : self.m + self.d, 0:1]  # After this, xv has dim [nsib, d, 1]

        # TODO Refactor this into sth cleaner
        v = self.stepInputMap(
            xv.view(xv.size(0), -1)
        )  # After this, q has dimensions [nsib, numel_output_mapper]
        v = torch.unsqueeze(
            v, dim=2
        )  # After this, q has dimensions [nsib, numel_output_mapper, 1]

        # print(f"xc = {xc}")
        # print(f"v = {v}")
        (
            self.cs.lc.A1,
            self.cs.lc.b1,
            self.cs.lc.A2,
            self.cs.lc.b2,
            self.cs.qcs.P,
            self.cs.qcs.P_sqrt,
            self.cs.qcs.q,
            self.cs.qcs.r,
            self.cs.socs.M,
            self.cs.socs.s,
            self.cs.socs.c,
            self.cs.socs.d,
            self.cs.lmis.F,
        ) = torch.vmap(self.constraintInputMap)(xc)
        ####################################################

        y = self.forwardForMethod(v)

        assert (
            torch.isnan(y).any()
        ) == False, f"If you are using DC3, try reducing args_DC3[lr]. Right now it's {self.args_DC3['lr']}"

        return y

    def gety0(self):
        return self.getyFromz(self.z0)

    def getyFromz(self, z):
        y = self.NA_E @ z + self.yp
        return y

    def getzFromy(self, y):
        z = self.NA_E.T @ (y - self.yp)
        return z

    def solveSecondOrderEq(self, a, b, c, is_quad_constraint):
        discriminant = torch.square(b) - 4 * (a) * (c)

        assert torch.all(
            discriminant >= 0
        ), f"Smallest element is {torch.min(discriminant)}"
        sol1 = torch.div(
            -(b) - torch.sqrt(discriminant), 2 * a
        )  # note that for quad constraints the positive solution has the minus: (... - sqrt(...))/(...)
        if is_quad_constraint:
            return sol1
        else:
            sol2 = torch.div(-(b) + torch.sqrt(discriminant), 2 * a)
            return torch.relu(torch.maximum(sol1, sol2))

    def isFeasible(self, y, eps=0.0):
        if self.cs.has_linear_ineq_constraints:
            violation = self.cs.lc.A1 @ y - self.cs.lc.b1 <= eps * torch.ones(
                self.cs.lc.b1.shape
            )
            utils.verify(
                torch.all(violation), "linear inequality constraints not satisfied"
            )

        if self.cs.has_linear_eq_constraints:
            violation = self.cs.lc.A2 @ y - self.cs.lc.b2
            utils.verify(
                torch.allclose(violation, torch.zeros(self.cs.lc.b2.shape), atol=eps),
                "linear equality constraints not satisfied",
            )

        if self.cs.has_quadratic_constraints:
            for i in range(self.cs.num_qc):
                idx = self.cs.qcs.at(i)
                violation = 0.5 * y.transpose(-1, -2) @ self.cs.qcs.P[
                    :, idx, :
                ] @ y + self.cs.qcs.q[:, idx, :].transpose(-1, -2) @ y + self.cs.qcs.r[
                    :, i, :
                ].unsqueeze(
                    1
                ) <= eps * torch.ones(
                    self.cs.qcs.r[:, i, :].unsqueeze(1).shape
                )
                utils.verify(
                    torch.all(violation), f"quadratic constraints {i} not satisfied"
                )

        if self.cs.has_soc_constraints:
            for i in range(self.cs.num_soc):
                idx = self.cs.socs.at(i)
                violation = torch.linalg.norm(
                    self.cs.socs.M[:, idx, :] @ y + self.cs.socs.s[:, idx, :], dim=1
                ).unsqueeze(1) - self.cs.socs.c[:, idx, :].transpose(
                    -1, -2
                ) @ y - self.cs.socs.d[
                    :, i, :
                ].unsqueeze(
                    1
                ) <= eps * torch.ones(
                    self.cs.socs.d[:, i, :].unsqueeze(1).shape
                )
                utils.verify(torch.all(violation), "SOC constraints not satisfied")

        if self.cs.has_lmi_constraints:
            lmi_left_hand_side = 0
            k = self.cs.lmis.dim
            for i in range(k):
                idx = self.cs.lmis.at(i)
                lmi_left_hand_side += (
                    y[:, i, :].unsqueeze(1) * self.cs.lmis.F[:, idx, :]
                )
            lmi_left_hand_side += self.cs.lmis.F[:, self.cs.lmis.at(k), :]
            eigenvalues = torch.linalg.eigvals(lmi_left_hand_side).real
            violation = eigenvalues >= -eps * torch.ones(eigenvalues.shape)
            utils.verify(torch.all(violation), "LMI constraints not satisfied")

    # def stackCholesky(self, A):
    #     output = torch.empty((0, self.k))
    #     for i in range(self.cs.num_qc):
    #         idx = self.cs.qcs.at(i)
    #         output = torch.cat((output, torch.linalg.cholesky(A[idx, :])), dim=0)
    #     return output


def nullSpace(batch_A):
    # A = torch.tensor(
    #     [[0.81649658, 0.0], [-0.40824829, -0.70710678], [-0.40824829, 0.70710678]]
    # )
    return torch_eye_like(batch_A.shape[0], batch_A.shape[2])


def torch_eye_like(batch_size, n):
    return torch.eye(n).unsqueeze(0).repeat(batch_size, 1, 1)
