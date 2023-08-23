# --------------------------------------------------------------------------
# Khai Nguyen | xkhai@cmu.edu
# Robotic Systems Lab, ETH Zürich
# Robotic Exploration Lab, Carnegie Mellon University
# See LICENSE file for the license information
# --------------------------------------------------------------------------

# Code inspired from https://github.com/locuslab/DC3/blob/main/method.py

import torch
import torch.nn as nn
import torch.optim as optim

import operator
from functools import reduce
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pickle
from tqdm import tqdm
import time
from datetime import datetime
import os
import subprocess
import argparse
import sys
from os.path import normpath, dirname, join

sys.path.insert(0, normpath(join(dirname(__file__), "../..")))

from rayen import constraints, constraint_module, utils
from examples.early_stopping import EarlyStopping

# pickle is lazy and does not serialize class definitions or function
# definitions. Instead it saves a reference of how to find the class
# (the module it lives in and its name)
from CbfQpProblem import CbfQpProblem

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# DEVICE = torch.device("cpu")
torch.set_default_device(DEVICE)
torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=4)

seed = 1999
torch.manual_seed(seed)
np.random.seed(seed)

print(f"{torch.get_num_threads() = }")
print(f"{torch.get_num_interop_threads() = }")


def main():
    utils.printInBoldBlue("CBF-QP Problem: Interior Point Network")
    print(f"{DEVICE = }")
    # Define problem
    args = {
        "xo": 1,
        "xc": 2,
        "nsamples": 12869,
        "epochs": 200,
        "batch_size": 256,
        "lr": 5e-6,
        "hidden_size": 128,
        "save_all_stats": True,  # otherwise, save latest stats only
        "res_save_freq": 5,
        "estop_patience": 5,
        "estop_delta": 0.01,  # improving rate of loss
        "seed": seed,
        "device": DEVICE,
        "board": True,
    }
    print(args)

    # Load data, and put on GPU if needed
    filepath = "data/cbf_qp_dataset2_xo{}_xc{}_ex{}".format(
        args["xo"], args["xc"], args["nsamples"]
    )

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    for attr in dir(data):
        var = getattr(data, attr)
        if not callable(var) and not attr.startswith("__") and torch.is_tensor(var):
            try:
                setattr(data, attr, var.to(args["device"]))
            except AttributeError:
                pass

    data._device = args["device"]
    dir_dict = {}

    TRAIN = 0

    if TRAIN:
        utils.printInBoldBlue("START TRAINING")
        dir_dict["now"] = datetime.now().strftime("%b%d_%H-%M-%S")
        dir_dict["save_dir"] = os.path.join(
            "results", "ipnn", str(data), dir_dict["now"]
        )
        dir_dict["tb_dir"] = os.path.join(
            "runs", "ipnn", dir_dict["now"] + "_" + str(data)
        )
        if not os.path.exists(dir_dict["save_dir"]):
            os.makedirs(dir_dict["save_dir"])
        with open(os.path.join(dir_dict["save_dir"], "args.dict"), "wb") as f:
            pickle.dump(args, f)

        train_net(data, args, dir_dict)
        print(f"{dir_dict['save_dir'] = }")
    else:
        utils.printInBoldBlue("START INFERENCE")
        dir_dict["infer_dir"] = os.path.join(
            "results", "ipnn", str(data), "Aug21_23-02-08", "model.dict"
        )
        infer_net(data, args, dir_dict)
    print(args)


def train_net(data, args, dir_dict=None):
    board = args["board"]
    if board:
        os.system("pkill -f tensorboard")
        # Set up TensorBoard
        writer = SummaryWriter(dir_dict["tb_dir"], flush_secs=1)
        # Find the latest run directory
        latest_run = os.listdir("runs/ipnn")[-1]
        # Start TensorBoard for the latest run
        subprocess.Popen(
            [
                f"python3 -m tensorboard.main --logdir={os.path.join('runs','ipnn')} --bind_all",
            ],
            shell=True,
        )

    # Some parameters
    solver_step = args["lr"]
    nepochs = args["epochs"]
    batch_size = args["batch_size"]

    # All data tensor
    dataset = TensorDataset(data.Xc, data.Y0)

    # Option 1
    # train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    #     dataset, [data.train_num, data.valid_num, data.test_num]
    # )

    ## Option 2: (already created with randomness). Keep the same order for infer_net
    train_dataset = torch.utils.data.Subset(dataset, range(data.train_num))
    valid_dataset = torch.utils.data.Subset(
        dataset, range(data.train_num, data.train_num + data.valid_num)
    )
    test_dataset = torch.utils.data.Subset(
        dataset,
        range(
            data.train_num + data.valid_num,
            data.train_num + data.valid_num + data.test_num,
        ),
    )
    print(f"{len(train_dataset) = }; {len(valid_dataset) = }; {len(test_dataset) = }")

    # To data batch
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device=args["device"]),
    )
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    # Network
    ip_net = IpNet(data, args)
    ip_net.to(args["device"])
    optimizer = optim.Adam(ip_net.parameters(), lr=solver_step)
    total_params = sum(p.numel() for p in ip_net.parameters())
    print(f"Number of parameters: {total_params}")

    earlyStopper = EarlyStopping(
        patience=args["estop_patience"], delta=args["estop_delta"], verbose=True
    )

    stats = {}  # statistics for all epochs

    # For each epoch
    for epoch in range(nepochs):
        epoch_stats = {}  # statistics for this epoach
        epoch_start_time = time.time()

        with torch.no_grad():
            # Get valid loss
            ip_net.eval()
            for valid_batch in valid_loader:
                Xvalid = valid_batch[0].to(args["device"])
                Yvalid = valid_batch[1].to(args["device"])
                eval_net(data, Xvalid, Yvalid, ip_net, args, "valid", epoch_stats)

            # Get test loss
            ip_net.eval()
            for test_batch in test_loader:
                Xtest = test_batch[0].to(args["device"])
                Ytest = test_batch[1].to(args["device"])
                eval_net(data, Xtest, Ytest, ip_net, args, "test", epoch_stats)

        # Get train loss
        ip_net.train()

        # for train_batch in tqdm(train_loader):
        for train_batch in train_loader:
            Xtrain = train_batch[0].to(args["device"])
            Ytrain = train_batch[1].to(args["device"]).squeeze(-1)
            start_time = time.time()
            optimizer.zero_grad(set_to_none=True)
            Yhat_train = ip_net(Xtrain)
            train_loss = total_loss(data, Xtrain, Ytrain, Yhat_train, args)
            train_loss.sum().backward()
            optimizer.step()
            train_time = time.time() - start_time
            # print(f"{train_time = }")
            dict_agg(epoch_stats, "train_loss", train_loss.detach().cpu().numpy())

        epoch_time = time.time() - epoch_start_time
        # print(f"{epoch_time = }")

        utils.printInBoldBlue(
            "Epoch {}: train loss {:.4f}, valid loss {:.4f}, test loss {:.4f}".format(
                epoch,
                np.mean(epoch_stats["train_loss"]),
                np.mean(epoch_stats["valid_loss"]),
                np.mean(epoch_stats["test_loss"]),
            )
        )

        # Print to TensorBoard
        if board:
            writer.add_scalars(
                "loss",
                {
                    "train": np.mean(epoch_stats["train_loss"]),
                    "valid": np.mean(epoch_stats["valid_loss"]),
                    "test": np.mean(epoch_stats["test_loss"]),
                },
                epoch,
            )
            writer.flush()

        # Log all statistics
        if args["save_all_stats"]:
            if epoch == 0:
                for key in epoch_stats.keys():
                    stats[key] = np.expand_dims(np.array(epoch_stats[key]), axis=0)
            else:
                for key in epoch_stats.keys():
                    stats[key] = np.concatenate(
                        (stats[key], np.expand_dims(np.array(epoch_stats[key]), axis=0))
                    )
        else:
            stats = epoch_stats

        # Early stop if not improving
        earlyStopper(
            np.mean(epoch_stats["valid_loss"]), ip_net, stats, dir_dict["save_dir"]
        )
        if earlyStopper.early_stop:
            utils.printInBoldRed("\nEarlyStopping: stop training!")
            utils.printInBoldGreen(
                "train loss {:.4f}, valid loss {:.4f}, test loss {:.4f}".format(
                    np.mean(epoch_stats["train_loss"]),
                    np.mean(epoch_stats["valid_loss"]),
                    np.mean(epoch_stats["test_loss"]),
                )
            )
            break

    if board:
        writer.close()
    return ip_net, stats


def total_loss(data, X, Y, Yhat, args):
    """Compute loss for batch X in supervised manner
    Don't use average here, only for evaluating
    Output: obj_cost (nsamples, 1)"""
    obj_cost = torch.square(Y - Yhat)
    return obj_cost


# Modifies stats in place
def dict_agg(stats, key, value, op="concat"):
    if key in stats.keys():
        if op == "sum":
            stats[key] += value
        elif op == "concat":
            stats[key] = np.concatenate((stats[key], value), axis=0)
        else:
            raise NotImplementedError
    else:
        stats[key] = value


@torch.no_grad()
def eval_net(data, X, Y, net, args, prefix, stats):
    make_prefix = lambda x: "{}_{}".format(prefix, x)
    start_time = time.time()
    Yhat = net(X)
    end_time = time.time()
    # print(f"{X=} {Y=}")
    dict_agg(
        stats,
        make_prefix("loss"),
        total_loss(data, X, Y, Yhat, args).detach().cpu().numpy(),
    )
    return stats


@torch.no_grad()
def infer_net(data, args, dir_dict=None):
    "Intuitvely evaluate random test data by inference"

    dataset = TensorDataset(data.Xc, data.Y0)
    test_dataset = torch.utils.data.Subset(
        dataset,
        range(
            data.train_num + data.valid_num,
            data.train_num + data.valid_num + data.test_num,
        ),
    )
    model = IpNet(data, args)
    model.load_state_dict(torch.load(dir_dict["infer_dir"]))
    model.eval()

    # Warm up the GPU
    X_dummy = torch.Tensor(500, data.xc_dim, 1).uniform_(-0.5, 0.5)
    _ = model(X_dummy.to(args["device"]))

    total_time = 0.0

    num = len(test_dataset)
    for i in range(50):
        idx = np.random.randint(0, num)
        X, Y0 = test_dataset[idx]
        X = X.unsqueeze(0)
        start_time = time.time()
        Ynn = model(X).item()
        total_time += time.time() - start_time
        Xo = X[0][0].item()
        # print(f"{Xo   = :.4f}")
        Y0 = Y0.item()
        utils.printInBoldGreen(f"{Y0 = :.4f}\n{Ynn  = :.4f}")
        print("--")

    infer_time = total_time / 50
    print(f"{infer_time=}")


###################################################################
# MODEL
###################################################################
class IpNet(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args

        # number of hidden layers and its size
        layer_sizes = [
            self._data.xc_dim,
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
            # nn.BatchNorm1d(layer_sizes[0]),
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            # nn.BatchNorm1d(layer_sizes[1]),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
            # nn.ReLU(),
            # nn.BatchNorm1d(layer_sizes[2]),
            nn.ReLU(),
            nn.Linear(layer_sizes[2], 1),
            nn.Dropout(p=0.1),
        ]

        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x is 3D
        return self.net(x.squeeze(-1))


if __name__ == "__main__":
    main()
    print()
