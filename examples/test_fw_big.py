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

import utils_examples
from nonfixed_examples import RppExample

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

example_number = 20
example = RppExample(example_number)

my_layer = constraint_module.ConstraintModule(
    xv_dim=example.xv_dim,
    xc_dim=example.xc_dim,
    y_dim=example.y_dim,
    method=method,
    num_cstr=example.num_cstr,
    cstrInputMap=example.cstrInputMap,
)

num_cstr_samples = 1

for i in range(num_cstr_samples):
    num_samples = 5

    # Define step input tensor
    xv_batched = torch.Tensor(num_samples, example.xv_dim).uniform_(-2, 2) * 150

    # Define constraint input tensor
    xc = torch.Tensor(example.xc_dim).uniform_(1, 5)
    # print(xc)
    # xc = torch.tensor([[10.0], [10.0]])
    xc_batched = xc.unsqueeze(0).repeat(num_samples, 1)
    x_batched = torch.cat((xv_batched, xc_batched), 1)

    # print(xv_batched)
    # print(xc_batched)
    # print(x_batched)

    # my_layer.eval()  # This changes the self.training variable of the module
    x_batched.requires_grad_()

    time_start = time.time()
    result = my_layer(x_batched)
    total_time_per_sample = (time.time() - time_start) / num_samples
    print(total_time_per_sample)
    # print(f"result = {result}")
    my_layer.isFeasible(result, 1e-8)

    result.sum().backward()

    # Call the autograd.gradcheck function
    correct_grad = torch.autograd.gradcheck(
        my_layer, x_batched, eps=1e-6, atol=1e-4, rtol=1e-3
    )

    # Print the result
    print("Gradient computation is correct:", correct_grad)
    # print("x_batched.grad = ", x_batched.grad)

    result = result.detach().numpy()

    y0 = my_layer.gety0()

    # print("FINISHED")
    # print(f"y0 = {y0}")
    # print(f"xv = {xv_batched}")
    # print(f"xc = {xc_batched}")
    # print(f"result = {result}")
