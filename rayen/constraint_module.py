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


class ConstraintModule(torch.nn.Module):
    def __init__(
        self,
        xv_dim=None,
        xc_dim=None,
        y_dim=None,
        method="RAYEN",
        args_DC3=None,
    ):
        super().__init__()

        self.method = method
        self.m = xv_dim  # Dimension of the step input vector xv
        self.d = xc_dim  # Dimension of the constraint input vector xc
        self.k = y_dim  # Dimension of the ambient space (output)
        self.n = None  # Dimension of the embedded space (determined later)
        self.batch_size = None

        self.has_linear_ineq_constraints = 1
        self.has_linear_eq_constraints = 0
        self.has_linear_constraints = (
            self.has_linear_eq_constraints or self.has_linear_ineq_constraints
        )
        self.has_quadratic_constraints = 0
        self.has_soc_constraints = 0
        self.has_lmi_constraints = 0

        self.selectSolver()

        # Pass dummy input to check the constraintInputMap users provide
        temp_x = torch.ones(xc_dim, 1)  # just a vector
        temp_A1, temp_b1, temp_A2, temp_b2 = constraintInputMap(temp_x)
        # Get the dimension of all potential constraints stacked together
        self.Ap_nrows = temp_A1.shape[0]

        if self.method == "RAYEN" or self.method == "RAYEN_old":
            # Handle dimensions of ambient and embedded space
            if self.has_linear_eq_constraints:
                self.n = self.k - 1  # Just one linear equality constraint
            else:
                self.n = self.k

            create_step_input_map = True if self.n != self.m else False

            self.setupInteriorPointLayer()

        if self.method == "RAYEN_old":
            self.forwardForMethod = self.forwardForRAYENOld
            self.dim_after_map = self.n + 1
        elif self.method == "RAYEN":
            self.forwardForMethod = self.forwardForRAYEN
            self.dim_after_map = self.n
        elif self.method == "UU":
            self.forwardForMethod = self.forwardForUU
            self.dim_after_map = self.k
        elif self.method == "Bar":
            self.forwardForMethod = self.forwardForBar
            self.dim_after_map = self.num_vertices + self.num_rays
        elif self.method == "PP":
            self.forwardForMethod = self.forwardForPP
            self.dim_after_map = self.n
        elif self.method == "UP":
            self.forwardForMethod = self.forwardForUP
            self.dim_after_map = self.n
        elif self.method == "DC3":
            self.forwardForMethod = self.forwardForDC3
            self.dim_after_map = self.k - self.neq_DC3
            assert self.dim_after_map == self.n
        else:
            raise NotImplementedError

        if create_step_input_map:
            self.stepInputMap = nn.Linear(self.m, self.n)
        else:
            self.stepInputMap = nn.Sequential()

    def selectSolver(self):
        installed_solvers = cp.installed_solvers()
        if ("GUROBI" in installed_solvers) and self.has_lmi_constraints == False:
            self.solver = "GUROBI"  # You need to do `python -m pip install gurobipy`
        elif ("ECOS" in installed_solvers) and self.has_lmi_constraints == False:
            self.solver = "ECOS"
        elif "SCS" in installed_solvers:
            self.solver = "SCS"
        # elif 'OSQP' in installed_solvers:
        # 	self.solver='OSQP'
        # elif 'CVXOPT' in installed_solvers:
        # 	self.solver='CVXOPT'
        else:
            # TODO: There are more solvers, see https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver
            raise Exception(f"Which solver do you have installed?")
            # Mapper does nothing
        self.solver = "SCS"
        return True

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

    def computeKappa(self, v_bar):
        kappa = torch.relu(torch.max(self.D @ v_bar, dim=1, keepdim=True).values)

        # if len(self.all_P) > 0 or len(self.all_M) > 0 or len(self.all_F) > 0:
        #     rho = self.NA_E @ v_bar
        #     rhoT = torch.transpose(rho, dim0=1, dim1=2)
        #     all_kappas_positives = torch.empty(
        #         (v_bar.shape[0], 0, 1), device=v_bar.device
        #     )

        #     for i in range(
        #         self.all_P.shape[0]
        #     ):  # for each of the quadratic constraints
        #         # FIRST WAY (slower, easier to understand)
        #         # P=self.all_P[i,:,:]
        #         # q=self.all_q[i,:,:]
        #         # r=self.all_r[i,:,:]

        #         # c_prime=0.5*rhoT@P@rho;
        #         # b_prime=(self.y0.T@P+ q.T)@rho;
        #         # a_prime=(0.5*self.y0.T@P@self.y0 + q.T@self.y0 +r)

        #         # kappa_positive_i_first_way=self.solveSecondOrderEq(a_prime, b_prime, c_prime, True)

        #         # SECOND WAY (faster)
        #         kappa_positive_i = self.all_phi[i, :, :] @ rho + torch.sqrt(
        #             rhoT @ self.all_delta[i, :, :] @ rho
        #         )

        #         # assert torch.allclose(kappa_positive_i,kappa_positive_i_first_way, atol=1e-06), f"{torch.max(torch.abs(kappa_positive_i-kappa_positive_i_first_way))}"

        #         assert torch.all(
        #             kappa_positive_i >= 0
        #         ), f"Smallest element is {kappa_positive_i}"  # If not, then Z may not be feasible (note that z0 is in the interior of Z)
        #         all_kappas_positives = torch.cat(
        #             (all_kappas_positives, kappa_positive_i), dim=1
        #         )

        #     for i in range(self.all_M.shape[0]):  # for each of the SOC constraints
        #         M = self.all_M[i, :, :]
        #         s = self.all_s[i, :, :]
        #         c = self.all_c[i, :, :]
        #         d = self.all_d[i, :, :]

        #         beta = M @ self.y0 + s
        #         tau = c.T @ self.y0 + d

        #         c_prime = rhoT @ M.T @ M @ rho - torch.square(c.T @ rho)
        #         b_prime = 2 * rhoT @ M.T @ beta - 2 * (c.T @ rho) @ tau
        #         a_prime = beta.T @ beta - torch.square(tau)

        #         kappa_positive_i = self.solveSecondOrderEq(
        #             a_prime, b_prime, c_prime, False
        #         )

        #         assert torch.all(
        #             kappa_positive_i >= 0
        #         )  # If not, then either the feasible set is infeasible (note that z0 is inside the feasible set)
        #         all_kappas_positives = torch.cat(
        #             (all_kappas_positives, kappa_positive_i), dim=1
        #         )

        #     if len(self.all_F) > 0:  # If there are LMI constraints:
        #         ############# OBTAIN S
        #         # First option (much slower)
        #         # S=self.all_F[0,:,:]*rho[:,0:(0+1),0:1]
        #         # for i in range(1,len(self.all_F)-1):
        #         # 	#See https://discuss.pytorch.org/t/scalar-matrix-multiplication-for-a-tensor-and-an-array-of-scalars/100174/2
        #         # 	S += self.all_F[i,:,:]*rho[:,i:(i+1),0:1]

        #         # Second option (much faster)
        #         S = torch.einsum(
        #             "ajk,ial->ijk", [self.all_F[0:-1, :, :], rho]
        #         )  # See the tutorial https://rockt.github.io/2018/04/30/einsum

        #         ############# COMPUTE THE EIGENVALUES

        #         ## Option 1: (compute whole spectrum of the matrix, using the non-symmetric matrix self.mHinv@S)
        #         # eigenvalues = torch.unsqueeze(torch.linalg.eigvals(self.mHinv@S),2) #Note that mHinv@M is not symmetric but always have real eigenvalues
        #         # assert (torch.all(torch.isreal(eigenvalues)))
        #         # largest_eigenvalue = torch.max(eigenvalues.real, dim=1, keepdim=True).values

        #         LTmSL = self.L.T @ (-S) @ self.L  # This matrix is symmetric

        #         ## Option 2: (compute whole spectrum of the matrix, using the symmetric matrix LTmSL). Much faster than Option 1
        #         eigenvalues = torch.unsqueeze(
        #             torch.linalg.eigvalsh(LTmSL), 2
        #         )  # Note that L^T (-S) L is a symmetric matrix
        #         largest_eigenvalue = torch.max(eigenvalues, dim=1, keepdim=True).values

        #         ## Option 3: Use LOBPCG with A=LTmSL and B=I. The advantage of this method is that only the largest eigenvalue is computed. But, empirically, this option is faster than option 2 only for very big matrices (>1000x1000)
        #         # guess_lobpcg=torch.rand(1, H.shape[0], 1);
        #         # size_batch=v_bar.shape[0]
        #         # largest_eigenvalue, _ = torch.lobpcg(A=LTmSL, k=1, B=None, niter=-1) #, X=guess_lobpcg.expand(size_batch, -1, -1)
        #         # largest_eigenvalue=torch.unsqueeze(largest_eigenvalue, 1)

        #         ## Option 4: Use power iteration to compute the largest eigenvalue. Often times is slower than just computing the whole spectrum, and sometimes it does not converge
        #         # guess_v = torch.nn.functional.normalize(torch.rand(S.shape[1],1), dim=0)
        #         # largest_eigenvalue=utils.findLargestEigenvalueUsingPowerIteration(self.mHinv@S, guess_v)

        #         ## Option 5: Use LOBPCG with A=-S and B=H. There are two problems though:
        #         # --> This issue: https://github.com/pytorch/pytorch/issues/101075
        #         # --> Backward is not implemented for B!=I, see: https://github.com/pytorch/pytorch/blob/d54fcd571af48685b0699f6ac1e31b6871d0d768/torch/_lobpcg.py#L329

        #         ## Option 6: Use https://github.com/rfeinman/Torch-ARPACK with LTmSL. The problem is that backward() is not implemented yet

        #         ## Option 7: Use https://github.com/buwantaiji/DominantSparseEigenAD. But it does not have support for batched matrices, see https://github.com/buwantaiji/DominantSparseEigenAD/issues/1

        #         kappa_positive_i = torch.relu(largest_eigenvalue)

        #         all_kappas_positives = torch.cat(
        #             (all_kappas_positives, kappa_positive_i), dim=1
        #         )

        #     kappa_nonlinear_constraints = torch.max(
        #         all_kappas_positives, dim=1, keepdim=True
        #     ).values
        #     kappa = torch.maximum(kappa, kappa_nonlinear_constraints)

        assert torch.all(kappa >= 0)

        return kappa

    # TODO
    # Function to recompute and register params
    def updateForwardParams(self):
        # self.register_buffer("A_p", torch.Tensor(self.cs.A_p))
        # self.register_buffer("b_p", torch.Tensor(self.cs.b_p))
        # self.register_buffer("yp", torch.Tensor(self.cs.yp))
        # self.register_buffer("NA_E", torch.Tensor(self.cs.NA_E))

        self.y0 = self.gety0()
        # print(f"y0 = {self.y0}")

        # self.cs.z0 = self.z0.detach().cpu().numpy()
        # self.cs.y0 = self.y0.detach().cpu().numpy()

        # Precompute for inverse distance to the frontier of Z along v_bar
        self.D = self.A_p / (
            (self.b_p - self.A_p @ self.z0) @ torch.ones(self.batch_size, 1, self.n)
        )  # for linear
        # print(f"D = {self.D}")
        # all_P, all_q, all_r = utils.getAllPqrFromQcs(self.cs.qcs)
        # all_M, all_s, all_c, all_d = utils.getAllMscdFromSocs(self.cs.socs)

        # if self.cs.has_lmi_constraints:
        #     all_F = copy.deepcopy(self.cs.lmic.all_F)
        #     H = all_F[-1]
        #     for i in range(self.cs.lmic.dim()):
        #         H += self.cs.y0[i, 0] * self.cs.lmic.all_F[i]
        #     Hinv = np.linalg.inv(H)
        #     mHinv = -Hinv
        #     L = np.linalg.cholesky(Hinv)  # Hinv = L @ L^T
        #     self.register_buffer("mHinv", torch.Tensor(mHinv))
        #     self.register_buffer("L", torch.Tensor(L))

        # else:
        #     all_F = []

        # # See https://discuss.pytorch.org/t/model-cuda-does-not-convert-all-variables-to-cuda/114733/9
        # # and https://discuss.pytorch.org/t/keeping-constant-value-in-module-on-correct-device/10129
        # self.register_buffer("D", torch.Tensor(D))
        # self.register_buffer("all_P", torch.Tensor(np.array(all_P)))
        # self.register_buffer("all_q", torch.Tensor(np.array(all_q)))
        # self.register_buffer("all_r", torch.Tensor(np.array(all_r)))
        # self.register_buffer("all_M", torch.Tensor(np.array(all_M)))
        # self.register_buffer("all_s", torch.Tensor(np.array(all_s)))
        # self.register_buffer("all_c", torch.Tensor(np.array(all_c)))
        # self.register_buffer("all_d", torch.Tensor(np.array(all_d)))
        # # self.register_buffer("all_F", torch.Tensor(np.array(all_F))) #This one dies (probably because out of memory) when all_F contains more than 7000 matrices 500x500 approx
        # self.register_buffer("all_F", torch.Tensor(all_F))

        # # self.register_buffer("z0", torch.Tensor(self.z0))
        # # self.register_buffer("y0", torch.Tensor(self.y0))

        # # Precompute to find roots of quadratic equation
        # if self.cs.has_quadratic_constraints:
        #     all_delta = []
        #     all_phi = []

        #     for i in range(
        #         self.all_P.shape[0]
        #     ):  # for each of the quadratic constraints
        #         P = self.all_P[i, :, :]
        #         q = self.all_q[i, :, :]
        #         r = self.all_r[i, :, :]
        #         y0 = self.y0

        #         sigma = 2 * (0.5 * y0.T @ P @ y0 + q.T @ y0 + r)
        #         phi = -(y0.T @ P + q.T) / sigma
        #         delta = (
        #             (y0.T @ P + q.T).T @ (y0.T @ P + q.T)
        #             - 4 * (0.5 * y0.T @ P @ y0 + q.T @ y0 + r) * 0.5 * P
        #         ) / torch.square(sigma)

        #         all_delta.append(delta)
        #         all_phi.append(phi)

        #     all_delta = torch.stack(all_delta)
        #     all_phi = torch.stack(all_phi)

        #     self.register_buffer("all_delta", all_delta)
        #     self.register_buffer("all_phi", all_phi)

        return True

    # TODO
    # From constraint input batch X, update all constraint data batch
    def updateSubspaceConstraints(self, cs_dict):
        if self.has_linear_constraints:
            # Retrive data from cs_dict
            A1 = cs_dict["A1"]
            b1 = cs_dict["b1"]
            A2 = cs_dict["A2"]
            b2 = cs_dict["b2"]

            # print(A1)
            # print(A2)
            # Stack the matrices so that the linear constraints look like Ax<=b
            if self.has_linear_ineq_constraints:
                A = A1
                b = b1
                if self.has_linear_eq_constraints:
                    # Add the equality constraints as inequality constraints
                    A = torch.cat((A, A2, -A2), axis=1)
                    b = torch.cat((b, b2, -b2), axis=1)
            else:
                # Add the equality constraints as inequality constraints
                A = torch.cat((A, A2, -A2), axis=1)
                b = torch.cat((b, b2, -b2), axis=1)

            print_debug_info = 0
            if print_debug_info:
                utils.printInBoldGreen(f"A is {A.shape} and b is {b.shape}")

            # TODO Preprocess
            # Here we simply choose E such that
            # A_E == A2, b_E == b_2
            # A_I == A1, b_I == b_1
            if self.has_linear_ineq_constraints:
                start = self.A1.shape[1]
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

            # if print_debug_info:
            #     utils.printInBoldGreen(f"AE={A_E}")
            #     utils.printInBoldGreen(f"AI={A_I}")

            # At this point, A_E, b_E, A_I, and b_I have at least one row

            # Project into the nullspace of A_E
            ################################################
            NA_E = nullSpace(A_E)
            # print(f"NA_E = {NA_E}")
            # n = NA_E.shape[2]  # dimension of the subspace

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
            # print(NA_E)
            yp = torch.zeros(self.batch_size, self.n, 1)
            # print(yp)
            # 0z<=1
            A_p = torch.zeros(self.batch_size, 1, self.n)
            # print(A_p)
            b_p = torch.ones(self.batch_size, 1, 1)
            # print(b_p)
            A_E = torch.zeros(self.batch_size, 1, self.n)
            # print(A_E)
            # 0y=0
            b_E = torch.zeros(self.batch_size, 1, 1)
            # print(b_E)
            A_I = torch.zeros(self.batch_size, 1, self.n)
            # print(A_I)
            # 0y<=1
            b_I = torch.ones(self.batch_size, 1, 1)
            # print(b_I)

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

        utils.verify(self.n == (self.k - np.linalg.matrix_rank(self.A_E[0])))
        return True

    # TODO
    def getConstraintsInSubspaceCvxpy(self, z, A_p, b_p, epsilon=0.0):
        constraints = [A_p @ z - b_p <= -epsilon * torch.ones((self.Ap_nrows, 1))]
        return constraints

    # TODO
    # Setup interior point problem beforehand
    def setupInteriorPointLayer(self):
        self.ip_epsilon = cp.Variable()
        self.ip_z0 = cp.Variable((self.n, 1))

        # self.ip_constraints = cp.Parameter((X, 1))  # need size
        A_p = cp.Parameter((self.Ap_nrows, self.n))
        b_p = cp.Parameter((self.Ap_nrows, 1))
        self.ip_constraints = self.getConstraintsInSubspaceCvxpy(
            self.ip_z0, A_p, b_p, self.ip_epsilon
        )
        # print(self.ip_constraints)
        self.ip_constraints.append(self.ip_epsilon >= 0)
        self.ip_constraints.append(
            self.ip_epsilon <= 0.5
        )  # This constraint is needed for the case where the set is unbounded. Any positive value is valid

        objective = cp.Minimize(-self.ip_epsilon)
        self.ip_prob = cp.Problem(objective, self.ip_constraints)

        assert self.ip_prob.is_dpp()
        self.ip_layer = CvxpyLayer(
            self.ip_prob,
            parameters=[A_p, b_p],
            variables=[self.ip_z0, self.ip_epsilon],
        )

        print_debug_info = 0
        if print_debug_info:
            utils.printInBoldBlue("Set up interior point problem")
        return True

    # TODO
    # Function to compute z0
    def solveInteriorPoint(self):
        # Update constraint
        # self.ip_constraints = self.cs.getConstraintsInSubspaceCvxpy(
        #     self.ip_z0, self.ip_epsilon
        # )
        # print(f"self.A_p = {self.A_p}")
        # print(f"self.b_p = {self.b_p}")
        ip_z0, ip_epsilon = self.ip_layer(
            self.A_p,
            self.b_p,
            solver_args={"solve_method": self.solver},
        )  # "max_iters": 10000
        # print(f"epsilon = {ip_epsilon}")
        # if ip_prob.status != "optimal" and ip_prob.status != "optimal_inaccurate":
        #     raise Exception(f"Value is not optimal, prob_status={ip_prob.status}")

        # utils.verify(
        #     epsilon.value > 1e-8
        # )  # If not, there are no strictly feasible points in the subspace

        return ip_z0

    # TODO
    # Forward pass for RAYEN
    def forwardForRAYEN(self, v, cs_dict):
        # Update Ap, bp, NA_E
        self.updateSubspaceConstraints(cs_dict)  # torch!!

        # Solve interior point
        self.z0 = self.solveInteriorPoint()
        # print(f"z0 = {self.z0}")

        # # Update and register all necessary parameters
        self.updateForwardParams()

        v_bar = torch.nn.functional.normalize(v, dim=1)
        kappa = self.computeKappa(v_bar)
        norm_v = torch.linalg.vector_norm(v, dim=(1, 2), keepdim=True)
        alpha = torch.minimum(1 / kappa, norm_v)
        return self.getyFromz(self.z0 + alpha * v_bar)

    def forward(self, x):
        ##################  MAPPER LAYER ####################
        # nn.Module forward method only accepts a single input tensor
        # nsib denotes the number of samples in the batch
        # each sample includes xv (size m) and xc (size d)
        # x has dimensions [nsib, m + d, 1]
        self.batch_size = x.shape[0]
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
        # TODO Refactor this into a class of ConvexConstraints
        self.A1, self.b1, self.A2, self.b2 = torch.vmap(constraintInputMap)(xc)
        cs_dict = {"A1": self.A1, "b1": self.b1, "A2": self.A2, "b2": self.b2}
        ####################################################

        y = self.forwardForMethod(v, cs_dict)

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


# TODO
# Problem-specific map from single example x to constraint data
def constraintInputMap(x):
    # x is a rank-2 tensor
    # outputs are rank-2 tensors
    # 2x3x1 @ 2x1x1 => 2x3x1
    A1 = torch.tensor(
        [
            [1.0, 0, 0],
            [0, 1.0, 0],
            [0, 0, 1.0],
            [-1.0, 0, 0],
            [0, -1.0, 0],
            [0, 0, -1.0],
        ]
    )
    b1 = torch.tensor([[1.0], [1.0], [1.0], [0], [0], [0]]) @ x
    A2 = torch.tensor([])
    b2 = torch.tensor([])
    # A2 = torch.tensor([[1.0, 1.0, 1.0]])
    # b2 = x[0, 0:1].unsqueeze(dim=1)
    return A1, b1, A2, b2  # ASK do I need to move it to device?


def nullSpace(batch_A):
    # A = torch.tensor(
    #     [[0.81649658, 0.0], [-0.40824829, -0.70710678], [-0.40824829, 0.70710678]]
    # )
    return torch_eye_like(batch_A.shape[0], batch_A.shape[2])


def torch_eye_like(batch_size, n):
    return torch.eye(n).unsqueeze(0).repeat(batch_size, 1, 1)
