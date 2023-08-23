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

    x0 = torch.Tensor([[[0.5], [-0.8]]])  # shape = (1, n, 1)
    v0 = torch.Tensor([[[0.5], [-0.3]]])  # shape = (1, n, 1)
    x0_n = x0.clone()
    v0_n = v0.clone()

    system = DoubleIntegrator(x0, v0, 1e-2)
    system_n = DoubleIntegrator(x0_n, v0_n, 1e-2)
    u_filtered = None
    un_filtered = None

    x_saved = np.empty((2, 1))
    v_saved = np.empty((2, 1))
    unom_saved = np.empty((2, 1))
    uf_saved = np.empty((2, 1))
    xn_saved = np.empty((2, 1))
    vn_saved = np.empty((2, 1))
    unf_saved = np.empty((2, 1))

    with torch.no_grad():
        for i in range(150):
            x, v = system.dynamics(u_filtered)
            xn, vn = system_n.dynamics(un_filtered)

            x_saved = np.concatenate((x_saved, x.numpy()[0]), axis=1)
            v_saved = np.concatenate((v_saved, v.numpy()[0]), axis=1)
            xn_saved = np.concatenate((xn_saved, xn.numpy()[0]), axis=1)
            vn_saved = np.concatenate((vn_saved, vn.numpy()[0]), axis=1)

            print(f"{x.squeeze() = }; {v.squeeze() = }")
            print(f"{xn.squeeze() = }; {vn.squeeze() = }")
            print(torch.norm(x - xn))

            # u_nom = torch.distributions.uniform.Uniform(-1, 1.0).sample(
            #     [1, args["xo"], 1]
            # )  # (1, n, 1)
            # u_nom = 1.5 * torch.tensor([[[np.cos(i / 30)], [np.sin(i / 30)]]])
            u_nom = torch.tensor([[[2.0], [-2.0]]])
            unom_saved = np.concatenate((unom_saved, u_nom.numpy()[0]), axis=1)

            un_filtered = nn_infer(model, xn, vn, u_nom)
            u_filtered = opt_solve(x, v, u_nom)

            unf_saved = np.concatenate((unf_saved, un_filtered.numpy()[0]), axis=1)
            uf_saved = np.concatenate((uf_saved, u_filtered.numpy()[0]), axis=1)

            print(f"{u_nom.squeeze() = }; {u_filtered.squeeze() = }")
            print(f"{u_nom.squeeze() = }; {un_filtered.squeeze() = } \n")

    # generate axes object
    ax = plt.subplot(3, 1, 1)
    plt.ylim(-1.2, 1.2)
    plt.xlabel("Timestep")
    plt.ylabel("Position")
    ax.plot(
        xn_saved[0, 1:],
        c="#7ccba2",
        linewidth=2,
        marker="o",
        markersize=4,
        label="pos xn",
    )
    ax.plot(
        xn_saved[1, 1:],
        c="#f0746e",
        linewidth=2,
        marker="o",
        markersize=4,
        label="pos yn",
    )
    ax.plot(x_saved[0, 1:], c="#045275", label="pos x", linewidth=2)
    ax.plot(x_saved[1, 1:], c="#7c1d6f", label="pos y", linewidth=2)
    plt.axhline(y=1, color="r", linestyle="dashed", label="upper limit")
    plt.axhline(y=-1, color="r", linestyle="dashed", label="lower limit")
    plt.legend()

    ax = plt.subplot(3, 1, 2)
    plt.ylim(-1.2, 1.2)
    plt.xlabel("Timestep")
    plt.ylabel("Velocity")
    ax.plot(
        vn_saved[0, 1:],
        c="#7ccba2",
        linewidth=2,
        marker="o",
        markersize=4,
        label="vel xn",
    )
    ax.plot(
        vn_saved[1, 1:],
        c="#f0746e",
        linewidth=2,
        marker="o",
        markersize=4,
        label="vel yn",
    )
    ax.plot(v_saved[0, 1:], c="#045275", label="vel x", linewidth=2)
    ax.plot(v_saved[1, 1:], c="#7c1d6f", label="vel y", linewidth=2)
    plt.axhline(y=1, color="r", linestyle="dashed", label="upper limit")
    plt.axhline(y=-1, color="r", linestyle="dashed", label="lower limit")
    plt.legend()

    ax = plt.subplot(3, 1, 3)
    plt.ylim(-1.2, 1.2)
    plt.xlabel("Timestep")
    plt.ylabel("Accel/Control")
    ax.plot(
        unf_saved[0, 1:],
        c="#7ccba2",
        linewidth=2,
        marker="o",
        markersize=4,
        label="accel xn",
    )
    ax.plot(
        unf_saved[1, 1:],
        c="#f0746e",
        linewidth=2,
        marker="o",
        markersize=4,
        label="accel yn",
    )
    ax.plot(uf_saved[0, 1:], c="#045275", label="accel x", linewidth=2)
    ax.plot(uf_saved[1, 1:], c="#7c1d6f", label="accel y", linewidth=2)
    plt.axhline(y=1, color="r", linestyle="dashed", label="upper limit")
    plt.axhline(y=-1, color="r", linestyle="dashed", label="lower limit")
    plt.legend()

    plt.show()


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
