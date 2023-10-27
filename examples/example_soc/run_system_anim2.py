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
torch.set_default_device(DEVICE)
torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=4)

seed = 1999
torch.manual_seed(seed)
np.random.seed(seed)


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
        "method": "RAYEN3",
        "training": False,
        "loss_type": "unsupervised",
        "epochs": 200,
        "batch_size": 64,
        "lr": 5e-3,
        "hidden_size": 64,
        "save_all_stats": True,  # otherwise, save latest stats only
        "res_save_freq": 5,
        "estop_patience": 10,
        "estop_delta": 0,  # improving rate of loss
        "seed": seed,
        "device": DEVICE,
        "board": False,
        "cbf_nn_dir": "Oct26_09-24-47",  # "Oct23_17-25-01",  # change
        "ip_nn_dir": "Oct23_15-12-34",  # keep this
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
    if (args["method"]) == "RAYEN1" or (args["method"]) == "RAYEN2":
        dir_dict["infer_dir"] = os.path.join(
            "results",
            "RAYEN1" + "_" + str(data),
            args["cbf_nn_dir"],
            "model.dict",
        )
    else:
        dir_dict["infer_dir"] = os.path.join(
            "results",
            "RAYEN3" + "_" + str(data),
            args["cbf_nn_dir"],
            "model.dict",
        )
    nn_layer.load_state_dict(torch.load(dir_dict["infer_dir"], map_location=DEVICE))

    if (args["method"]) == "RAYEN2" or (args["method"]) == "RAYEN3":
        ip_nn = nn.Sequential(
            nn.Linear(args["xc"], args["hidden_size"]),
            nn.BatchNorm1d(args["hidden_size"]),
            nn.ReLU(),
            nn.Linear(args["hidden_size"], args["hidden_size"]),
            nn.BatchNorm1d(args["hidden_size"]),
            nn.ReLU(),
            nn.Linear(args["hidden_size"], args["xo"]),
        )
        # Load ip_nn
        dir_dict["ip_nn_dir"] = os.path.join(
            "results", "ipnn", str(data), args["ip_nn_dir"], "model.dict"
        )
        ip_nn.load_state_dict(torch.load(dir_dict["ip_nn_dir"], map_location=DEVICE))
        ip_nn.eval()
        print("ip_nn loaded")
    else:
        ip_nn = None

    cbf_net = constraint_module2.ConstraintModule(
        args["xo"],
        args["xc"],
        args["xo"],
        args["method"],
        ip_nn,
        data.num_cstr,
        data.cstrInputMap,
        nn_layer,
    )
    cbf_net.eval()

    x0 = torch.Tensor([[[0.0], [0.0], [0.0]]])  # shape = (1, n, 1)
    v0 = torch.Tensor([[[0.1], [0.1], [-0.2]]])  # shape = (1, n, 1)
    x0_n = x0.clone()
    v0_n = v0.clone()

    system = DoubleIntegrator(x0, v0, 1e-2)
    system_n = DoubleIntegrator(x0_n, v0_n, 1e-2)
    u_opt = None
    u_nn = None

    # plt.draw()
    # plt.pause(5)
    # utils.printInBoldBlue("STARTTTTTTTTTTTTTTTTTTT")
    # plt.pause(3.5)
    stats = Statistics()
    steps = 200
    with torch.no_grad():
        for i in range(steps):
            x, v = system.dynamics(u_opt)
            xn, vn = system_n.dynamics(u_nn)

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

            u_nn, time_nn = nn_infer(cbf_net, xn, vn, u_nom)
            u_opt, time_opt, time_opt2 = opt_solve(x, v, u_nom)
            stats._all_time_nn.append(time_nn)
            stats._all_time_opt.append(time_opt)
            stats._all_time_opt2.append(time_opt2)
            stats._all_obj_nn.append(torch.norm(u_nn - u_nom).numpy())
            stats._all_obj_opt.append(torch.norm(u_opt - u_nom).numpy())
            # assert torch.norm(u_filtered) < 1.01
            # assert torch.norm(un_filtered) < 1.01 and torch.norm(u_filtered) < 1.01

            verbose = 0
            if verbose:
                utils.printInBoldBlue(f"step = {i}")
                print(f"{x.squeeze() = }; {v.squeeze() = }")
                print(f"{xn.squeeze() = }; {vn.squeeze() = }")
                print(f"{u_nom.squeeze() = }; {u_opt.squeeze() = }")
                print(f"{u_nom.squeeze() = }; {u_nn.squeeze() = } \n")

            # draw_scene(un_filtered, u_filtered, u_nom, i)

    stats.compute()
    print(stats)


def nn_infer(model, xn, vn, u_nom):
    input_n = torch.cat((u_nom, xn, vn), dim=1)
    start_time = time.time()
    un_filtered = model(input_n)
    infer_time = time.time() - start_time
    # print(f"inference time = {infer_time}")
    un_filtered.nelement() == 0 and utils.printInBoldRed("NN failed")
    return un_filtered, infer_time


def opt_solve(x, v, u_nom):
    xc = torch.cat([u_nom, x, v], 1).squeeze(-1)
    problem = CbfSocProblem(xc, 3, 6, 3)
    problem.updateObjective()
    problem.updateConstraints()
    start_time = time.time()
    _, opt_time = problem.computeY(tol=1e-2, max_iters=1000)
    opt_time2 = time.time() - start_time
    u_filtered = problem.Y
    u_filtered.nelement() == 0 and utils.printInBoldRed("Solver failed")
    return u_filtered.unsqueeze(-1), opt_time, opt_time2


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


class Statistics:
    def __init__(self):
        self._all_obj_nn = []
        self._mean_obj_nn = 0.0
        self._std_obj_nn = 0.0
        self._all_obj_opt = []
        self._mean_obj_opt = 0.0
        self._std_obj_opt = 0.0
        self._all_time_nn = []
        self._mean_time_nn = 0.0
        self._std_time_nn = 0.0
        self._all_time_opt = []
        self._mean_time_opt = 0.0
        self._std_time_opt = 0.0
        self._all_time_opt2 = []
        self._mean_time_opt2 = 0.0
        self._std_time_opt2 = 0.0

    def compute(self):
        # Compute mean and std
        self._mean_obj_nn = np.mean(
            np.array(self._all_obj_nn) / np.array(self._all_obj_opt)
        )
        import pdb

        # pdb.set_trace()
        self._std_obj_nn = np.std(
            np.array(self._all_obj_nn) / np.array(self._all_obj_opt)
        )
        self._mean_obj_opt = np.mean(self._all_obj_opt)
        self._std_obj_opt = np.std(self._all_obj_opt)
        self._mean_time_nn = np.mean(self._all_time_nn)
        self._std_time_nn = np.std(self._all_time_nn)
        self._mean_time_opt = np.mean(self._all_time_opt)
        self._std_time_opt = np.std(self._all_time_opt)
        self._mean_time_opt2 = np.mean(self._all_time_opt2)
        self._std_time_opt2 = np.std(self._all_time_opt2)

    def __repr__(self) -> str:
        return (
            f"mean_l2_nn    = {self._mean_obj_nn:.4f} ({self._std_obj_nn:.4f})\n"
            f"mean_l2_opt   = {self._mean_obj_opt:.4f} ({self._std_obj_opt:.4f}) \n"
            f"mean_time_nn  = {self._mean_time_nn:.4f} ({self._std_time_nn:.4f}) \n"
            f"mean_time_opt2 = {self._mean_time_opt2:.4f} ({self._std_time_opt2:.4f}) \n"
            f"mean_time_opt = {self._mean_time_opt:.8f} ({self._std_time_opt:.4f}) \n"
        )


def draw_scene(un_filtered, u_filtered, u_nom, i):
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


if __name__ == "__main__":
    main()
    print()
