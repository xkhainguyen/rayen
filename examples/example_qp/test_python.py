import torch
import torch.nn as nn
import torch.optim as optim

import operator
from functools import reduce
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pickle
import time
import os
import argparse

import sys
from os.path import normpath, dirname, join

sys.path.insert(0, normpath(join(dirname(__file__), "..")))

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_dtype(torch.float64)
torch.set_default_device(DEVICE)

with open(
    "/home/khai/SSD/Code/rayen/examples/example_qp/results/QpProblem-1-2-18368/Aug16_09-31-47/args.dict",
    "rb",
) as f:
    data = pickle.load(f)

print(data)

# A1 = torch.tensor([[[0, 0], [0, 0]], [[1, 2], [3, 4.]]])
# print(A1.dtype)
# b1 = torch.tensor([[[2], [1]], [[2], [1.]]])
# # Yhat = torch.tensor([])
# Apinv = torch.linalg.pinv(A1)
# print(Apinv)

# lstsq = torch.linalg.lstsq(A1, b1)
# yp = lstsq.solution
# print(f"{lstsq.residuals = }")

# import cvxpy as cp
# import numpy as np
# import torch
# from cvxpylayers.torch import CvxpyLayer

# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# # DEVICE = torch.device("cpu")
# print(f"{DEVICE=}")
# torch.set_default_device(DEVICE)
# torch.set_default_dtype(torch.float64)
# np.set_printoptions(precision=4)

# x = cp.Variable((2, 1))
# xc = cp.Parameter(2)
# xc.value = np.array([1.0, 2.5])
# xc_torch = torch.tensor([1.0, 2.5])
# obj = cp.Minimize((x[0] - xc[0]) ** 2 + (x[1] - xc[1]) ** 2)
# A = torch.tensor([[1, -2.0], [-1, -2], [-1, 2]], device="cpu")
# # A = np.array([[1, -2.0], [-1, -2], [-1, 2]])
# b = torch.tensor([[2], [6], [2.0]], device="cuda").cpu()
# # b = np.array([[2], [6], [2.0]])
# cons = [A @ x + b >= np.array([[0.0], [0.0], [0]])]
# # cons = [
# #     (x[0] - 2 * x[1] + 2) >= 0,
# #     (-x[0] - 2 * x[1] + 6) >= 0,
# #     (-x[0] + 2 * x[1] + 2) >= 0,
# # ]
# prob = cp.Problem(obj, cons)
# cvxpylayer = CvxpyLayer(problem=prob, parameters=[xc], variables=[x])

# prob.solve(solver="ECOS")

# for i in prob.variables():
#     print(i.value)

# z = cvxpylayer(xc_torch, solver_args={"solve_method": "ECOS"})
# print(z)
