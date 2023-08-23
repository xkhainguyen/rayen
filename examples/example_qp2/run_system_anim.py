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
from CbfQpProblem import CbfQpProblem

# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("cpu")
# torch.set_default_device(DEVICE)
torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=8)

# # generate axes object
# ax = plt.axes()
# ax.set_box_aspect(1)
# rect = patches.Rectangle((-1, -1), 2, 2, color="red", fill=False)
# ax.add_patch(rect)
# # set limits
# plt.xlim(-1.2, 1.2)
# plt.ylim(-1.2, 1.2)
# plt.xlabel("Position x")
# plt.ylabel("Position y")


def main():
    utils.printInBoldBlue("CBF-QP Problem")
    print(f"{DEVICE = }")
    # Define problem
    args = {
        "prob_type": "cbf_qp",
        "xo": 2,
        "xc": 4,
        "nsamples": 9591,
        "method": "RAYEN",
        "hidden_size": 32,
    }
    print(args)

    # Load data, and put on GPU if needed
    prob_type = args["prob_type"]
    if prob_type == "cbf_qp":
        filepath = "data/cbf_qp_dataset2_xo{}_xc{}_ex{}".format(
            args["xo"], args["xc"], args["nsamples"]
        )
    else:
        raise NotImplementedError

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    data._device = DEVICE
    dir_dict = {}

    utils.printInBoldBlue("START INFERENCE")
    dir_dict["infer_dir"] = os.path.join(
        "results", str(data), "Aug22_22-36-27", "model.dict"
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

    x0 = torch.Tensor([[[0.9], [0.9]]])  # shape = (1, n, 1)
    v0 = torch.Tensor([[[0.1], [0.1]]])  # shape = (1, n, 1)
    x0_n = x0.clone()
    v0_n = v0.clone()

    system = DoubleIntegrator(x0, v0, 1e-2)
    system_n = DoubleIntegrator(x0_n, v0_n, 1e-2)
    u_filtered = None
    un_filtered = None
    with torch.no_grad():
        for i in range(150):
            x, v = system.dynamics(u_filtered)
            xn, vn = system_n.dynamics(un_filtered)
            print(f"{x.squeeze() = }; {v.squeeze() = }")
            print(f"{xn.squeeze() = }; {vn.squeeze() = }")
            print(torch.norm(x - xn))

            # u_nom = torch.distributions.uniform.Uniform(-1, 1.0).sample(
            #     [1, args["xo"], 1]
            # )  # (1, n, 1)
            u_nom = 1.5 * torch.tensor([[[np.cos(i / 30)], [np.sin(i / 30)]]])
            # u_nom = torch.tensor([[[2.0], [2.0]]])

            un_filtered = nn_infer(model, xn, vn, u_nom)
            u_filtered = opt_solve(x, v, u_nom)

            print(f"{u_nom.squeeze() = }; {u_filtered.squeeze() = }")
            print(f"{u_nom.squeeze() = }; {un_filtered.squeeze() = } \n")

            # # add something to axes
            # ax.scatter(xn.squeeze()[0], xn.squeeze()[1], s=100.0, c="orange")
            # ax.quiver(
            #     xn.squeeze()[0],
            #     xn.squeeze()[1],
            #     u_nom.squeeze()[0],
            #     u_nom.squeeze()[1],
            #     scale=10,
            #     color="orange",
            # )
            # ax.scatter(x.squeeze()[0], x.squeeze()[1], s=100.0, c="blue", alpha=0.5)
            # ax.quiver(
            #     x.squeeze()[0],
            #     x.squeeze()[1],
            #     u_nom.squeeze()[0],
            #     u_nom.squeeze()[1],
            #     scale=10,
            #     color="blue",
            #     alpha=0.5,
            # )

            # # draw the plot
            # plt.draw()
            # plt.pause(0.1)  # is necessary for the plot to update for some reason

            # # start removing points if you don't want all shown
            # if i > 0:
            #     ax.collections[0].remove()

            #     ax.collections[1].remove()
            #     # plt.legend(["nn", "opt"], loc=2)
            #     ax.collections[2].remove()

            #     ax.collections[3].remove()
            #     plt.legend(["limit", "nn", "", "opt", ""], loc=2)


def nn_infer(model, xn, vn, u_nom):
    input_n = torch.cat((u_nom, xn, vn), dim=1)
    un_filtered = model(input_n)
    un_filtered.nelement() == 0 and utils.printInBoldRed("NN failed")
    return un_filtered


def opt_solve(x, v, u_nom):
    xc = torch.cat([u_nom, x, v], 1).squeeze(-1)
    problem = CbfQpProblem(xc, 2, 4, 2)
    problem.updateObjective()
    problem.updateConstraints()
    problem.computeY()
    u_filtered = problem.Y
    u_filtered.nelement() == 0 and utils.printInBoldRed("Solver failed")
    return u_filtered.unsqueeze(-1)


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
