# --------------------------------------------------------------------------
# Khai Nguyen | xkhai@cmu.edu
# Robotic Systems Lab, ETH Zürich
# Robotic Exploration Lab, Carnegie Mellon University
# See LICENSE file for the license information
# --------------------------------------------------------------------------

# Code inspired https://github.com/locuslab/DC3/blob/main/datasets/simple/make_dataset.py
# This class stores the data data and solve optimization

import numpy as np
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader

import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))

from CbfSocProblem import CbfSocProblem
from rayen import utils, constraint_module, constraints_torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

DEVICE = torch.device("cpu")
torch.set_default_device(DEVICE)
torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=4)

seed = 1999
torch.manual_seed(seed)
np.random.seed(seed)


if __name__ == "__main__":
    num_samples = 15000
    xo_dim = 3  # nominal control u_bar dimension
    y_dim = 3  # filtered control, output of the network
    pos_dim = 3
    vel_dim = 3
    xc_dim = pos_dim + vel_dim  # state x dimension

    args = {
        "prob_type": "cbf_soc",
        "xo": xo_dim,
        "xc": xc_dim,
        "nsamples": num_samples,
        "device": DEVICE,
    }

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
                setattr(data, attr, var.to(args["device"]))
            except AttributeError:
                pass

    data.updateConstraints()

    layer = constraint_module.ConstraintModule(
        data.y_dim,
        data.xc_dim,
        data.y_dim,
        "RAYEN",
        data.num_cstr,
        data.cstrInputMap,
    )
    # Xo = torch.tensor([[[2.1]], [[2.0]]])
    # Xc = torch.tensor([[[1.0], [0.8]], [[1.0], [0.8]]])
    # xv can arbitrary
    Y = layer(data.Xo.squeeze(-1), data.Xc.squeeze(-1))  # 3D
    # Y = layer(Xo, Xc)
    Y0 = layer.z0  # 3D
    print(f"{Y0[0] = }")
    print(f"{layer.isFeasible(Y0, 1e-2)}")  # Just test feasibility

    with open(
        "./data/ip_dataset_xo{}_xc{}_ex{}".format(xo_dim, xc_dim, data.nsamples),
        "wb",
    ) as f:
        pickle.dump(data, f)
