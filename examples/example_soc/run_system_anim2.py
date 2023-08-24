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
from CbfSocProblem import CbfSocProblem

# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("cpu")
# torch.set_default_device(DEVICE)
torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=4)


def soc(x, y):
    return np.sqrt(x**2 + y**2) / 5


# SOC
u_soc, v_soc = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
x_soc = np.cos(u_soc) * np.sin(v_soc)
y_soc = np.sin(u_soc) * np.sin(v_soc)
z_soc = soc(x_soc, y_soc)

# Sphere
radius = 1
theta = np.linspace(0, 2.0 * np.pi, 40)
phi = np.linspace(0, np.pi, 40)
x_sphere = radius * np.outer(np.cos(theta), np.sin(phi))
y_sphere = radius * np.outer(np.sin(theta), np.sin(phi))
z_sphere = radius * np.outer(np.ones(np.size(theta)), np.cos(phi))

# Plot option
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("accel control x")
ax.set_ylabel("accel control y")
ax.set_zlabel("accel control z")
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)
ax.set_zlim(-1, 1)
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=4, azim=-66)


def draw_quiver(p1, p2, color, **kwargs):
    len = torch.norm(p2).numpy()
    ax.quiver(
        p1.squeeze()[0],
        p1.squeeze()[1],
        p1.squeeze()[2],
        p2.squeeze()[0],
        p2.squeeze()[1],
        p2.squeeze()[2],
        # scale=20,
        length=len,
        color=color,
        **kwargs,
    )


def main():
    utils.printInBoldBlue("CBF-SOC Problem")
    print(f"{DEVICE = }")
    # Define problem
    args = {
        "prob_type": "cbf_soc",
        "xo": 3,
        "xc": 6,
        "nsamples": 15000,
        "method": "RAYEN",
        "loss_type": "unsupervised",
        "epochs": 100,
        "batch_size": 64,
        "lr": 5e-3,
        "hidden_size": 64,
        "save_all_stats": True,  # otherwise, save latest stats only
        "res_save_freq": 5,
        "estop_patience": 5,
        "estop_delta": 0,  # improving rate of loss
        "device": DEVICE,
        "board": False,
    }
    print(args)

    # Load data, and put on GPU if needed
    prob_type = args["prob_type"]
    if prob_type == "cbf_soc":
        filepath = "data/cbf_soc_dataset_xo{}_xc{}_ex{}".format(
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
        "results", str(data), "Aug23_22-31-40", "model.dict"
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

    x0 = torch.Tensor([[[0.0], [0.0], [0.0]]])  # shape = (1, n, 1)
    v0 = torch.Tensor([[[0.1], [0.1], [-0.2]]])  # shape = (1, n, 1)
    x0_n = x0.clone()
    v0_n = v0.clone()

    system = DoubleIntegrator(x0, v0, 1e-2)
    system_n = DoubleIntegrator(x0_n, v0_n, 1e-2)
    u_filtered = None
    un_filtered = None

    steps = 400
    # plt.draw()
    # plt.pause(5)
    # utils.printInBoldBlue("STARTTTTTTTTTTTTTTTTTTT")
    # plt.pause(3.5)
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
            u_nom = 0.5 * torch.tensor(
                [
                    [
                        [2.3 * np.cos(i / 20)],
                        [2.3 * np.sin(i / 20)],
                        [0.5 * np.sin(i / 20) + 0.5],
                    ]
                ]
            )
            # u_nom = torch.tensor([[[0.1], [0.1], [-0.01]]])

            un_filtered, time_nn = nn_infer(model, xn, vn, u_nom)
            u_filtered, time_opt = opt_solve(x, v, u_nom)
            time_nn_sum += time_nn
            time_opt_sum += time_opt
            # assert torch.norm(u_filtered) < 1.01
            # assert torch.norm(un_filtered) < 1.01 and torch.norm(u_filtered) < 1.01

            print(f"{u_nom.squeeze() = }; {u_filtered.squeeze() = }")
            print(f"{u_nom.squeeze() = }; {un_filtered.squeeze() = } \n")

            # add something to axes
            size = 200.0
            ax.scatter(
                un_filtered.squeeze()[0],
                un_filtered.squeeze()[1],
                un_filtered.squeeze()[2],
                s=size,
                c="#f0746e",
                marker="*",
            )
            ax.scatter(
                u_filtered.squeeze()[0],
                u_filtered.squeeze()[1],
                u_filtered.squeeze()[2],
                s=size,
                c="#7ccba2",
                marker="*",
                alpha=1,
            )
            ax.scatter(
                u_nom.squeeze()[0],
                u_nom.squeeze()[1],
                u_nom.squeeze()[2],
                s=size,
                c="gray",
                marker="*",
                alpha=1,
            )
            ax.plot_surface(x_soc, y_soc, z_soc, color="#3c93c2", alpha=0.05)
            ax.plot_surface(x_sphere, y_sphere, z_sphere, color="#fcde9c", alpha=0.04)
            # draw the plot
            plt.draw()
            plt.pause(0.0001)  # is necessary for the plot to update for some reason
            # start removing points if you don't want all shown
            if i > 0:
                [ax.collections[0].remove() for _ in range(5)]
                plt.legend(["u_nn", "u_opt", "u_nom"], loc=2)

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
    problem = CbfSocProblem(xc, 3, 6, 3)
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
