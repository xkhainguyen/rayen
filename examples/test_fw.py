# --------------------------------------------------------------------------
# Jesus Tordesillas Torres, Robotic Systems Lab, ETH Zürich
# See LICENSE file for the license information
# --------------------------------------------------------------------------

import numpy as np
import random
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
import waitGPU

import utils_examples
from nonfixed_examples import RppExample

import fixpath  # Following this example: https://github.com/tartley/colorama/blob/master/demos/demo01.py
from rayen import constraints, constraint_module, utils

np.set_printoptions(precision=2)

# Set the default device to GPU if available, otherwise use CPU
device = "cpu"
torch.set_default_tensor_type(
    torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor
)
torch.set_default_dtype(torch.float64)

seed = 1999
torch.manual_seed(seed)
np.random.seed(seed)

method = "RAYEN"

example_number = 13
example = RppExample(example_number)

my_layer = constraint_module.ConstraintModule(
    xv_dim=example.xv_dim,
    xc_dim=example.xc_dim,
    y_dim=example.y_dim,
    method=method,
    num_cstr=example.num_cstr,
    cstrInputMap=example.cstrInputMap,
).to(device)

fig = plt.figure(figsize=(6, 6))
fig.suptitle(method + ": " + example.name, fontsize=14)

num_cstr_samples = 1
rows = math.ceil(math.sqrt(num_cstr_samples))
cols = rows

for i in range(num_cstr_samples):
    num_samples = 1000

    # Define step input tensor
    xv_batched_x = torch.Tensor(num_samples, 1).uniform_(-2, 2) * 150
    xv_batched_y = torch.Tensor(num_samples, 1).uniform_(-2, 2) * 150
    if example.xv_dim == 3:
        xv_batched_z = torch.Tensor(num_samples, 1).uniform_(-2, 2) * 150
        xv_batched = torch.cat((xv_batched_x, xv_batched_y, xv_batched_z), 1)
    if example.xv_dim == 2:
        xv_batched = torch.cat((xv_batched_x, xv_batched_y), 1)
    # print(xv_batched)
    # xv = torch.tensor([0.0, 10.5, 1.0])
    # xv_batched = xv.unsqueeze(0).repeat(num_samples, 1)

    # Define constraint input tensor
    # xc = torch.Tensor(example.xc_dim).uniform_(1, 5)
    # print(xc)
    xc = torch.tensor([10.2, 10.1, 0.0, 3.0])
    xc_batched = xc.unsqueeze(0).repeat(num_samples, 1)
    # x_batched = torch.cat((xv_batched, xc_batched), 1)

    print(xv_batched.device)
    # print(xc_batched)
    # print(x_batched)

    my_layer.eval()  # This changes the self.training variable of the module

    time_start = time.time()
    result = my_layer(xv_batched, xc_batched)
    total_time_per_sample = (time.time() - time_start) / num_samples
    print(total_time_per_sample)
    # print(f"result = {result}")
    my_layer.isFeasible(result, 1e-8)

    result = result.detach().cpu().numpy()
    xc = xc.cpu()

    y0 = my_layer.gety0().cpu()

    # print("FINISHED")
    # print(f"y0 = {y0}")
    # print(f"xv = {xv_batched}")
    # print(f"xc = {xc_batched}")
    # print(f"result = {result}")

    ax = fig.add_subplot(
        rows, cols, i + 1, projection="3d" if example.y_dim == 3 else None
    )
    ax.set_title(f"xc = {xc.numpy().T}")
    ax.title.set_size(10)
    ax.set_aspect("equal", "box")
    if example.y_dim == 2:
        ax.scatter(y0[0, 0, 0], y0[0, 1, 0], color="r", s=100)
        ax.scatter(result[:, 0, 0], result[:, 1, 0])
    if example.y_dim == 3:
        ax.scatter(y0[0, 0, 0], y0[0, 1, 0], y0[0, 2, 0], color="r", s=100)
        ax.scatter(result[:, 0, 0], result[:, 1, 0], result[:, 2, 0])

plt.show()
