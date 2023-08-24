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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("Vel x")
    ax.set_ylabel("Vel y")
    ax.set_zlabel("Vel z")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)

    x0 = torch.Tensor([[[0.0], [0.0], [0.0]]])  # shape = (1, n, 1)
    v0 = torch.Tensor([[[0.5], [0.5], [0.5]]])  # shape = (1, n, 1)
    x0_n = x0.clone()
    v0_n = v0.clone()

    system = DoubleIntegrator(x0, v0, 1e-2)
    system_n = DoubleIntegrator(x0_n, v0_n, 1e-2)
    u_filtered = None
    un_filtered = None
    with torch.no_grad():
        for i in range(100):
            x, v = system.dynamics(u_filtered)
            xn, vn = system_n.dynamics(un_filtered)
            print(f"{x.squeeze() = }; {v.squeeze() = }")
            print(f"{xn.squeeze() = }; {vn.squeeze() = }")

            # print(torch.norm(v - vn))

            # u_nom = torch.distributions.uniform.Uniform(-1, 1.0).sample(
            #     [1, args["xo"], 1]
            # )  # (1, n, 1)
            # u_nom = 2 * torch.tensor([[[np.cos(i / 20)], [np.sin(i / 20)]]])
            u_nom = torch.tensor([[[2.0], [0.0], [0.0]]])

            un_filtered = nn_infer(model, xn, vn, u_nom)
            u_filtered = opt_solve(x, v, u_nom)
            print(torch.norm(u_filtered))
            # assert torch.norm(u_filtered) < 1.01
            # assert torch.norm(un_filtered) < 1.01 and torch.norm(u_filtered) < 1.01

            print(f"{u_nom.squeeze() = }; {u_filtered.squeeze() = }")
            print(f"{u_nom.squeeze() = }; {un_filtered.squeeze() = } \n")

            # add something to axes
            def draw_quiver(p1, p2, color, **kwargs):
                len = torch.norm(p1 - p2).numpy()
                ax.quiver(
                    p1.squeeze()[0],
                    p1.squeeze()[1],
                    p1.squeeze()[2],
                    p2.squeeze()[0],
                    p2.squeeze()[1],
                    p2.squeeze()[0],
                    # scale=20,
                    length=len / 5,
                    color=color,
                    **kwargs,
                )

            draw_quiver(vn, u_nom, "gray", alpha=0.2)
            draw_quiver(v, u_nom, "gray", alpha=0.2)

            # nn green
            ax.scatter(
                vn.squeeze()[0], vn.squeeze()[1], vn.squeeze()[2], s=100.0, c="#7ccba2"
            )
            draw_quiver(vn, un_filtered, "#7ccba2")

            # opt orange
            ax.scatter(
                v.squeeze()[0],
                v.squeeze()[1],
                v.squeeze()[2],
                s=100.0,
                c="#f0746e",
                alpha=1,
            )
            draw_quiver(v, u_filtered, "#f0746e", alpha=1)

            # draw the plot
            plt.draw()
            plt.pause(0.2)  # is necessary for the plot to update for some reason

            # start removing points if you don't want all shown
            if i > 0:
                # n = len(ax.collections)
                ax.collections[0].remove()
                ax.collections[0].remove()
                # plt.legend(["nn", "opt"], loc=2)
                ax.collections[0].remove()
                ax.collections[0].remove()
                ax.collections[0].remove()
                ax.collections[0].remove()
                # plt.legend(["limit", "opt", "", "nn", ""], loc=2)


def nn_infer(model, xn, vn, u_nom):
    input_n = torch.cat((u_nom, xn, vn), dim=1)
    start_time = time.time()
    un_filtered = model(input_n)
    infer_time = time.time() - start_time
    print(f"inference time = {infer_time}")
    un_filtered.nelement() == 0 and utils.printInBoldRed("NN failed")
    return un_filtered


def opt_solve(x, v, u_nom):
    xc = torch.cat([u_nom, x, v], 1).squeeze(-1)
    problem = CbfSocProblem(xc, 3, 6, 3)
    problem.updateObjective()
    problem.updateConstraints()
    problem.computeY(tol=1e-2)
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


###################################################################
# MODEL
###################################################################
class CbfQcqpNet(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args

        # number of hidden layers and its size
        layer_sizes = [
            self._data.x_dim,
            self._args["hidden_size"],
            self._args["hidden_size"],
            self._args["hidden_size"],
        ]
        # layers = reduce(
        #     operator.add,
        #     [
        #         # [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU()]
        #         [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.1)]
        #         for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])
        #     ],
        # )

        layers = [
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            # nn.BatchNorm1d(layer_sizes[1]),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
            nn.ReLU(),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
        ]

        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

        self.nn_layer = nn.Sequential(*layers)

        self.rayen_layer = constraint_module.ConstraintModule(
            layer_sizes[-1],
            self._data.xc_dim,
            self._data.y_dim,
            self._args["method"],
            self._data.num_cstr,
            self._data.cstrInputMap,
        )

    def forward(self, x):
        x = x.squeeze(-1)
        xv = self.nn_layer(x)
        xc = x[:, self._data.xo_dim : self._data.xo_dim + self._data.xc_dim]
        y = self.rayen_layer(xv, xc)
        return y


if __name__ == "__main__":
    main()
    print()
