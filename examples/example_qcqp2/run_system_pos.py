# --------------------------------------------------------------------------
# Khai Nguyen | xkhai@cmu.edu
# Robotic Systems Lab, ETH Zürich
# Robotic Exploration Lab, Carnegie Mellon University
# See LICENSE file for the license information
# --------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim

import operator
from functools import reduce
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt, patches
from matplotlib.animation import FuncAnimation

import numpy as np
import pickle
import time
from datetime import datetime
import os
import subprocess
import argparse
import sys
from os.path import normpath, dirname, join

sys.path.insert(0, normpath(join(dirname(__file__), "../..")))

from rayen import constraints, constraint_module2, utils
from examples.early_stopping import EarlyStopping

# pickle is lazy and does not serialize class definitions or function
# definitions. Instead it saves a reference of how to find the class
# (the module it lives in and its name)
from CbfQcqpProblem import CbfQcqpProblem

# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("cpu")
# torch.set_default_device(DEVICE)
torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=4)

# generate axes object
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_box_aspect(1)

# set limits
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
circle1 = patches.Circle((0.0, 0.0), radius=1.0, color="black", fill=False)
ax.add_patch(circle1)


def main():
    utils.printInBoldBlue("CBF-QCQP Problem")
    print(f"{DEVICE = }")
    # Define problem
    args = {
        "prob_type": "cbf_qcqp",
        "xo": 2,
        "xc": 4,
        "nsamples": 15405,
        "method": "RAYEN",
        "hidden_size": 64,
    }
    print(args)

    # Load data, and put on GPU if needed
    prob_type = args["prob_type"]
    if prob_type == "cbf_qcqp":
        filepath = "data/cbf_qcqp_dataset_xo{}_xc{}_ex{}".format(
            args["xo"], args["xc"], args["nsamples"]
        )
    else:
        raise NotImplementedError

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    for attr in dir(data):
        var = getattr(data, attr)
        if not callable(var) and not attr.startswith("__") and torch.is_tensor(var):
            try:
                setattr(data, attr, var.to(DEVICE))
            except AttributeError:
                pass

    data._device = DEVICE
    dir_dict = {}

    utils.printInBoldBlue("START INFERENCE")
    dir_dict["infer_dir"] = os.path.join(
        "results", str(data), "Aug24_12-12-12", "model.dict"
    )
    # MODEL
    nn_layer = nn.Sequential(
        nn.Linear(args["xo"] + args["xc"] + args["xo"], args["hidden_size"]),
        nn.BatchNorm1d(args["hidden_size"]),
        nn.ReLU(),
        nn.Linear(args["hidden_size"], args["hidden_size"]),
        nn.BatchNorm1d(args["hidden_size"]),
        nn.ReLU(),
        nn.Linear(args["hidden_size"], args["xo"]),
    )

    model = constraint_module2.ConstraintModule(
        args["xo"],
        args["xc"],
        args["xo"],
        args["method"],
        data.num_cstr,
        data.cstrInputMap,
        nn_layer,
    )
    model.load_state_dict(torch.load(dir_dict["infer_dir"]))
    model.eval()

    time_opt_sum = 0
    time_nn_sum = 0

    x0 = torch.Tensor([[[0.0], [0.0]]])  # shape = (1, n, 1)
    v0 = torch.Tensor([[[-0.92], [0.0]]])  # shape = (1, n, 1)
    x0_n = torch.Tensor([[[0.0], [0.0]]])  # shape = (1, n, 1)
    v0_n = torch.Tensor([[[-0.92], [0.0]]])  # shape = (1, n, 1)

    system = DoubleIntegrator(x0, v0, 1e-2)
    system_n = DoubleIntegrator(x0_n, v0_n, 1e-2)
    u_filtered = None
    un_filtered = None
    steps = 200
    plt.draw()
    plt.pause(10)
    utils.printInBoldBlue("STARTTTTTTTTTTTTTTTTTTT")
    plt.pause(3.5)

    with torch.no_grad():
        for i in range(steps):
            utils.printInBoldBlue(f"step = {i}")
            x, v = system.dynamics(u_filtered)
            xn, vn = system_n.dynamics(un_filtered)
            print(f"{x.squeeze() = }; {v.squeeze() = }")
            print(f"{xn.squeeze() = }; {vn.squeeze() = }")

            # print(torch.norm(v - vn))

            # u_nom = torch.distributions.uniform.Uniform(-1, 1.0).sample(
            #     [1, args["xo"], 1]
            # )  # (1, n, 1)
            # u_nom = 2 * torch.tensor([[[np.cos(i / 20)], [np.sin(i / 20)]]])
            u_nom = torch.tensor([[[0.0], [-1.1]]])

            un_filtered, time_nn = nn_infer(model, xn, vn, u_nom)
            u_filtered, time_opt = opt_solve(x, v, u_nom)

            time_nn_sum += time_nn
            time_opt_sum += time_opt

            print(torch.norm(u_filtered))
            # assert torch.norm(u_filtered) < 1.01
            # assert torch.norm(un_filtered) < 1.01 and torch.norm(u_filtered) < 1.01

            print(f"{u_nom.squeeze() = }; {u_filtered.squeeze() = }")
            print(f"{u_nom.squeeze() = }; {un_filtered.squeeze() = } \n")

            ###############
            ## DRAW VELOCITY
            ###############
            # plt.xlabel("vel x")
            # plt.ylabel("vel y")
            size = 200.0
            ax.scatter(vn.squeeze()[0], vn.squeeze()[1], s=size, c="#f0746e")
            ax.scatter(v.squeeze()[0], v.squeeze()[1], s=size, c="#7ccba2", alpha=0.5)

            ax.quiver(
                vn.squeeze()[0],
                vn.squeeze()[1],
                u_nom.squeeze()[0],
                u_nom.squeeze()[1],
                scale=10,
                color="gray",
                alpha=0.5,
            )
            ax.quiver(
                v.squeeze()[0],
                v.squeeze()[1],
                u_nom.squeeze()[0],
                u_nom.squeeze()[1],
                scale=10,
                color="gray",
                alpha=0.5,
            )
            # draw the plot
            plt.draw()
            plt.pause(0.1)  # is necessary for the plot to update for some reason

            # start removing points if you don't want all shown
            plt.xlabel("x")
            plt.ylabel("y")
            if i > 0:
                [ax.collections[0].remove() for _ in range(4)]
                plt.legend(["limit", "v_nn", "v_opt", "u_nom"], loc=2)

    plt.draw()
    plt.pause(5)
    print(f"average nn time = {time_nn_sum/steps}")
    print(f"average opt time = {time_opt_sum/steps}")


def nn_infer(model, xn, vn, u_nom):
    input_n = torch.cat((u_nom, xn, vn), dim=1)
    start_time = time.time()
    un_filtered = model(input_n)
    infer_time = time.time() - start_time
    print(f"inference time = {infer_time}")
    un_filtered.nelement() == 0 and utils.printInBoldRed("NN failed")
    return un_filtered, infer_time


def opt_solve(x, v, u_nom):
    xc = torch.cat([u_nom, x, v], 1).squeeze(-1)
    problem = CbfQcqpProblem(xc, 2, 4, 2)
    problem.updateObjective()
    problem.updateConstraints()
    start_time = time.time()
    problem.computeY(tol=1e-2)
    opt_time = time.time() - start_time
    u_filtered = problem.Y
    u_filtered.nelement() == 0 and utils.printInBoldRed("Solver failed")
    return u_filtered.unsqueeze(-1), opt_time


###################################################################
# SYSTEM
###################################################################
class DoubleIntegrator:
    def __init__(self, x0, v0, dt):
        self._x = x0
        self._v = v0
        self._dt = dt
        self._t = 0.0

    @property
    def x(self):
        return self._x

    @property
    def v(self):
        return self._v

    @property
    def dt(self):
        return self._dt

    @property
    def t(self):
        return self._t

    def dynamics(self, u=None):
        if u is not None:
            self._v += u * self.dt
        self._x += self._v * self.dt
        self._t += self.dt
        return self.x, self.v


if __name__ == "__main__":
    main()
    print()
