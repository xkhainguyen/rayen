# --------------------------------------------------------------------------
# Jesus Tordesillas Torres, Robotic Systems Lab, ETH Zürich
# See LICENSE file for the license information
# --------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import numpy as np
import scipy
import os
import time

from examples_sets import getExample
import utils_examples

import fixpath  # Following this example: https://github.com/tartley/colorama/blob/master/demos/demo01.py
from rayen import constraints, constraint_module, utils

np.set_printoptions(precision=2)

torch.set_default_dtype(torch.float64)
# Set the default device to GPU if available, otherwise use CPU
# device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.set_default_tensor_type(
#     torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor
# )

method = "RAYEN"

fig = plt.figure()
fig.suptitle(method, fontsize=14)

# tmp = getExample(index_example)

# constraint = constraints.ConvexConstraints(
#     lc=tmp.lc,
#     qcs=tmp.qcs,
#     socs=tmp.socs,
#     lmic=tmp.lmic,
#     # y0=np.array([[0.5], [0.0], [0.8]]),
# )


# TODO
# Problem-specific map from single example x to constraint data
def constraintInputMap(x):
    # x is a rank-2 tensor
    # outputs are rank-2 tensors

    # Linear constraints
    A1 = torch.tensor([])
    b1 = torch.tensor([])
    A2 = torch.tensor([])
    b2 = torch.tensor([])

    # Quadratic constraints
    r1 = 1.0
    c1 = x
    E1 = (1 / (r1 * r1)) * torch.eye(2)
    P1 = 2 * E1
    q1 = -2 * E1 @ c1
    r1 = c1.transpose(-1, -2) @ E1 @ c1 - 1

    r2 = 1.0
    c2 = x * 0.9
    E2 = (1 / (r2 * r2)) * torch.eye(2)
    P2 = 2 * E2
    q2 = -2 * E2 @ c2
    r2 = c2.transpose(-1, -2) @ E2 @ c2 - 1

    P = torch.cat((P1, P2), dim=0)
    q = torch.cat((q1, q2), dim=0)
    r = torch.cat((r1, r2), dim=0)

    return A1, b1, A2, b2, P, q, r


# # TODO
# # Problem-specific map from single example x to constraint data
# def constraintInputMap(x):
#     # x is a rank-2 tensor
#     # outputs are rank-2 tensors
#     # 2x3x1 @ 2x1x1 => 2x3x1
#     A1 = torch.tensor(
#         [
#             [1.0, 0, 0],
#             [0, 1.0, 0],
#             [0, 0, 1.0],
#             [-1.0, 0, 0],
#             [0, -1.0, 0],
#             [0, 0, -1.0],
#         ]
#     )
#     b1 = torch.tensor([[1.0], [1.0], [1.0], [0], [0], [0]]) @ x
#     A2 = torch.tensor([])
#     b2 = torch.tensor([])
#     # A2 = torch.tensor([[1.0, 1.0, 1.0]])
#     # b2 = x[0, 0:1].unsqueeze(dim=1)
#     return A1, b1, A2, b2  # ASK do I need to move it to device?

y_dim = 2
xc_dim = 2
num_cstr = [0, 0, 2, 0, 0]  # linear ineq, linear eq, qcs, socs, lmis
my_layer = constraint_module.ConstraintModule(
    2, xc_dim, y_dim, method=method, num_cstr, constraintInputMap=constraintInputMap
)

num_cstr_samples = 1
rows = math.ceil(math.sqrt(num_cstr_samples))
cols = rows

for i in range(num_cstr_samples):
    num_samples = 300

    # Define step input tensor
    xv_batched_x = torch.Tensor(num_samples, 1, 1).uniform_(-2, 2)
    xv_batched_y = torch.Tensor(num_samples, 1, 1).uniform_(-2, 2)
    # xv_batched_z = torch.Tensor(num_samples, 1, 1).uniform_(-2.5, 2.5)
    xv_batched = torch.cat((xv_batched_x, xv_batched_y), 1)
    # print(xv_batched)
    # xv_batched = torch.tensor([[[10.0], [10.0], [10.0]]])

    # xc_batched = torch.tensor([[1.0]]).unsqueeze(-1).repeat(num_samples, 1, 1)
    # xc_batched = torch.tensor([[[1.0]]])

    # Define constraint input tensor
    # xc = torch.Tensor(xc_dim, 1).uniform_(1, 3)
    xc = torch.tensor([[10.0], [10.0]])  # FIXME why does this need origin for IP = 0
    xc_batched = xc.unsqueeze(0).repeat(num_samples, 1, 1)
    x_batched = torch.cat((xv_batched, xc_batched), 1)

    # print(xv_batched)
    # print(xc_batched)
    # print(x_batched)

    my_layer.eval()  # This changes the self.training variable of the module

    time_start = time.time()
    result = my_layer(x_batched)
    total_time_per_sample = (time.time() - time_start) / num_samples

    result = result.detach().numpy()

    y0 = my_layer.gety0()

    # print("FINISHED")
    # print(f"y0 = {y0}")
    # print(f"xv = {xv_batched}")
    # print(f"xc = {xc_batched}")
    # print(f"result = {result}")

    ax = fig.add_subplot(rows, cols, i + 1, projection="3d" if y_dim == 3 else None)
    ax.set_title(f"xc = {xc.numpy().T}")
    ax.title.set_size(10)
    ax.scatter(y0[0, 0, 0], y0[0, 1, 0], color="r", s=100)
    ax.scatter(result[:, 0, 0], result[:, 1, 0])


# my_dict = constraint.getDataAsDict()
# my_dict["result"] = result
# my_dict["total_time_per_sample"] = total_time_per_sample
# my_dict["y0"] = constraint.y0
# my_dict["v"] = v_batched.detach().numpy()
# directory = "./first_figure"
# if not os.path.exists(directory):
#     os.makedirs(directory)
# scipy.io.savemat(directory + "/first_figure.mat", my_dict)

# utils.printInBoldBlue(
#     f"Example {index_example}, total_time_per_sample={total_time_per_sample}"
# )

plt.show()
