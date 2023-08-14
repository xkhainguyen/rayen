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
import time
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

# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("cpu")
print(f"{DEVICE=}")
torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=4)

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)


def main():
    # Define problem
    args = {
        "prob_type": "cbf_qp",
        "xo": 1,
        "xc": 2,
        "nsamples": 953,
        "method": "RAYEN",
        "loss_type": "unsupervised",
        "epochs": 100,
        "batch_size": 100,
        "lr": 2e-6,
        "hidden_size": 500,
        "save_all_stats": True,  # otherwise, save latest stats only
        "res_save_freq": 5,
        "estop_patience": 5,
        "estop_delta": 0.05,  # improving rate of loss
    }

    # Load data, and put on GPU if needed
    prob_type = args["prob_type"]
    if prob_type == "cbf_qp":
        filepath = "data/cbf_qp_dataset_xo{}_xc{}_ex{}".format(
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

    TRAIN = 1

    if TRAIN:
        save_dir = os.path.join(
            "results",
            str(data),
            str(time.time()).replace(".", "-"),
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "args.dict"), "wb") as f:
            pickle.dump(args, f)
        train_net(data, args, save_dir)
        print(f"{save_dir=}")
    else:
        infer_dir = os.path.join(
            "results", str(data), "1692016320-6447008", "cbf_qp_net.dict"
        )
        infer_net(data, args, infer_dir)


def train_net(data, args, save_dir=None):
    # Set up TensorBoard
    writer = SummaryWriter(flush_secs=1)
    # Find the latest run directory
    latest_run = os.listdir("runs")[-1]
    # Start TensorBoard for the latest run
    subprocess.Popen(
        [
            f"python3 -m tensorboard.main --logdir={os.path.join('runs', latest_run)}",
        ],
        shell=True,
    )

    # Some parameters
    solver_step = args["lr"]
    nepochs = args["epochs"]
    batch_size = args["batch_size"]

    # All data tensor
    dataset = TensorDataset(data.X, data.Y)

    ## First option
    # train_dataset = TensorDataset(data.trainX)
    # valid_dataset = TensorDataset(data.validX)
    # test_dataset = TensorDataset(data.testX)

    ## Second option
    # train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    #     dataset, [data.train_num, data.valid_num, data.test_num]
    # )

    ## Third option (already created with randomness). Keep the same order for infer_net
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
    print(f"{len(train_dataset)=}")
    print(f"{len(valid_dataset)=}")
    print(f"{len(test_dataset)=}")

    # To data batch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    # Network
    cbf_qp_net = CbfQpNet(data, args)
    cbf_qp_net.to(DEVICE)
    optimizer = optim.Adam(cbf_qp_net.parameters(), lr=solver_step)

    earlyStopper = EarlyStopping(
        patience=args["estop_patience"], delta=args["estop_delta"], verbose=True
    )

    stats = {}  # statistics for all epochs

    # For each epoch
    for i in range(nepochs):
        epoch_stats = {}  # statistics for this epoach

        with torch.no_grad():
            # Get valid loss
            cbf_qp_net.eval()
            for valid_batch in valid_loader:
                Xvalid = valid_batch[0].to(DEVICE)
                Yvalid = valid_batch[1].to(DEVICE)
                eval_net(data, Xvalid, Yvalid, cbf_qp_net, args, "valid", epoch_stats)

            # Get test loss
            cbf_qp_net.eval()
            for test_batch in test_loader:
                Xtest = test_batch[0].to(DEVICE)
                Ytest = test_batch[1].to(DEVICE)
                eval_net(data, Xtest, Ytest, cbf_qp_net, args, "test", epoch_stats)

        # Get train loss
        cbf_qp_net.train()

        for train_batch in train_loader:
            Xtrain = train_batch[0].to(DEVICE)
            Ytrain = train_batch[1].to(DEVICE).unsqueeze(-1)
            start_time = time.time()
            optimizer.zero_grad()
            Yhat_train = cbf_qp_net(Xtrain)
            train_loss = total_loss(data, Xtrain, Ytrain, Yhat_train, args)
            train_loss.sum().backward()
            optimizer.step()
            train_time = time.time() - start_time
            dict_agg(epoch_stats, "train_loss", train_loss.detach().cpu().numpy())

        utils.printInBoldBlue(
            "Epoch {}: train loss {:.4f}, valid loss {:.4f}, test loss {:.4f}".format(
                i,
                np.mean(epoch_stats["train_loss"]),
                np.mean(epoch_stats["valid_loss"]),
                np.mean(epoch_stats["test_loss"]),
            )
        )

        # Print to TensorBoard
        writer.add_scalars(
            "loss",
            {
                "train": np.mean(epoch_stats["train_loss"]),
                "valid": np.mean(epoch_stats["valid_loss"]),
                "test": np.mean(epoch_stats["test_loss"]),
            },
            i,
        )
        writer.flush()

        # Log all statistics
        if args["save_all_stats"]:
            if i == 0:
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
        earlyStopper(np.mean(epoch_stats["valid_loss"]), cbf_qp_net, stats, save_dir)
        if earlyStopper.early_stop:
            utils.printInBoldRed("EarlyStopping: stop training!")
            break

    writer.close()
    return cbf_qp_net, stats


def total_loss(data, X, Y, Yhat, args):
    """Compute loss for batch X in supervised or unsupervised manner
    Output: obj_cost (nsamples, 1)"""

    if args["loss_type"] == "supervised":
        obj_cost = torch.square(Y - Yhat)
    else:
        Xo = data.getXo(X)
        data.updateObjective(Xo)
        obj_cost = data.objectiveFunction(Yhat)
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
def infer_net(data, args, save_dir):
    "Evaluate random test data intuitively"

    dataset = TensorDataset(data.X, data.Y)
    test_dataset = torch.utils.data.Subset(
        dataset,
        range(
            data.train_num + data.valid_num,
            data.train_num + data.valid_num + data.test_num,
        ),
    )
    model = CbfQpNet(data, args)
    model.load_state_dict(torch.load(save_dir))
    model.eval()

    for i in range(10):
        idx = np.random.randint(0, len(test_dataset))
        X, Y = dataset[idx]
        print(f"{X.T =}")
        X = X.unsqueeze(0)
        Yhat = model(X).item()
        Y = Y.item()
        print(f"{Y   =:.4f}\n{Yhat=:.4f}")
        print("--")


###################################################################
# MODEL
###################################################################
class CbfQpNet(nn.Module):
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
        layers = reduce(
            operator.add,
            [
                # [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU()]
                [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.1)]
                for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])
            ],
        )

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
    os.system("pkill -f tensorboard")
    main()
