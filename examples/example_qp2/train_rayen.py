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

from rayen import constraints, constraint_module2, utils
from examples.early_stopping import EarlyStopping

# pickle is lazy and does not serialize class definitions or function
# definitions. Instead it saves a reference of how to find the class
# (the module it lives in and its name)
from CbfQpProblem import CbfQpProblem

# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("cpu")
torch.set_default_device(DEVICE)
torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=4)

seed = 1999
torch.manual_seed(seed)
np.random.seed(seed)

print(f"{torch.get_num_threads() = }")
print(f"{torch.get_num_interop_threads() = }")


def main():
    utils.printInBoldBlue("CBF-QP Problem")
    print(f"{DEVICE = }")
    # Define problem
    args = {
        "prob_type": "cbf_qp",
        "xo": 12,
        "xc": 24,
        "nsamples": 25045,
        "method": "RAYEN",
        "loss_type": "unsupervised",
        "epochs": 500,
        "batch_size": 256,
        "lr": 5e-3,
        "hidden_size": 256,
        "save_all_stats": True,  # otherwise, save latest stats only
        "res_save_freq": 5,
        "estop_patience": 20,
        "estop_delta": 0,  # improving rate of loss
        "seed": seed,
        "device": DEVICE,
        "board": False,
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

    for attr in dir(data):
        var = getattr(data, attr)
        if not callable(var) and not attr.startswith("__") and torch.is_tensor(var):
            try:
                setattr(data, attr, var.to(args["device"]))
            except AttributeError:
                pass

    data._device = args["device"]
    dir_dict = {}

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

    cbf_qp_net = constraint_module2.ConstraintModule(
        args["xo"],
        args["xc"],
        args["xo"],
        args["method"],
        data.num_cstr,
        data.cstrInputMap,
        nn_layer,
    )

    TRAIN = 1

    if TRAIN:
        utils.printInBoldBlue("START TRAINING")
        dir_dict["now"] = datetime.now().strftime("%b%d_%H-%M-%S")
        dir_dict["save_dir"] = os.path.join("results", str(data), dir_dict["now"])
        dir_dict["tb_dir"] = os.path.join("runs", dir_dict["now"] + "_" + str(data))
        if not os.path.exists(dir_dict["save_dir"]):
            os.makedirs(dir_dict["save_dir"])
        with open(os.path.join(dir_dict["save_dir"], "args.dict"), "wb") as f:
            pickle.dump(args, f)

        train_net(cbf_qp_net, data, args, dir_dict)
        print(f"{dir_dict['save_dir'] = }")
    else:
        utils.printInBoldBlue("START INFERENCE")
        dir_dict["infer_dir"] = os.path.join(
            "results", str(data), "Aug22_20-53-33", "model.dict"
        )
        infer_net(cbf_qp_net, data, args, dir_dict)
    print(args)


def train_net(cbf_qp_net, data, args, dir_dict=None):
    board = args["board"]
    if board:
        os.system("pkill -f tensorboard")
        # Set up TensorBoard
        writer = SummaryWriter(dir_dict["tb_dir"], flush_secs=1)
        # Find the latest run directory
        latest_run = os.listdir("runs")[-1]
        # Start TensorBoard for the latest run
        subprocess.Popen(
            [
                f"python3 -m tensorboard.main --logdir={os.path.join('runs')} --bind_all",
                # f"python3 -m tensorboard.main --logdir={os.path.join('runs', latest_run)} --bind_all",
            ],
            shell=True,
        )

    # Some parameters
    solver_step = args["lr"]
    nepochs = args["epochs"]
    batch_size = args["batch_size"]

    # All data tensor
    dataset = TensorDataset(data.X, data.Y, data.obj_val, data.Y0)

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

    # Compute ground truth averaged objective/loss
    train_truth_obj = compute_truth_obj(train_dataset)
    valid_truth_obj = compute_truth_obj(valid_dataset)
    test_truth_obj = compute_truth_obj(test_dataset)

    print(f"{train_truth_obj = }; {valid_truth_obj = }; {test_truth_obj = }")

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
    cbf_qp_net.to(args["device"])
    optimizer = optim.Adam(cbf_qp_net.parameters(), lr=solver_step)
    total_params = sum(p.numel() for p in cbf_qp_net.parameters())
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
            cbf_qp_net.eval()
            for valid_batch in valid_loader:
                Xvalid = valid_batch[0].to(args["device"])
                Yvalid = valid_batch[1].to(args["device"])
                Y0valid = valid_batch[3].to(args["device"])
                cbf_qp_net.z0 = Y0valid
                eval_net(data, Xvalid, Yvalid, cbf_qp_net, args, "valid", epoch_stats)

            # Get test loss
            cbf_qp_net.eval()
            for test_batch in test_loader:
                Xtest = test_batch[0].to(args["device"])
                Ytest = test_batch[1].to(args["device"])
                Y0test = test_batch[3].to(args["device"])
                cbf_qp_net.z0 = Y0test
                eval_net(data, Xtest, Ytest, cbf_qp_net, args, "test", epoch_stats)

        # Get train loss
        cbf_qp_net.train()

        # for train_batch in tqdm(train_loader):
        for train_batch in train_loader:
            Xtrain = train_batch[0].to(args["device"])
            Ytrain = train_batch[1].to(args["device"]).unsqueeze(-1)
            start_time = time.time()
            optimizer.zero_grad(set_to_none=True)
            Y0train = train_batch[3].to(args["device"])
            cbf_qp_net.z0 = Y0train
            Yhat_train = cbf_qp_net(Xtrain)
            train_loss = total_loss(data, Xtrain, Ytrain, Yhat_train, args)
            train_loss.sum().backward()
            optimizer.step()
            train_time = time.time() - start_time
            # print(f"{train_time = }")
            dict_agg(epoch_stats, "train_loss", train_loss.detach().cpu().numpy())

        epoch_time = time.time() - epoch_start_time
        # print(f"{epoch_time = }")

        utils.printInBoldBlue(
            "Epoch {}: train loss {:.4f}/{:.4f}, valid loss {:.4f}/{:.4f}, test loss {:.4f}/{:.4f}".format(
                epoch,
                np.mean(epoch_stats["train_loss"]),
                train_truth_obj,
                np.mean(epoch_stats["valid_loss"]),
                valid_truth_obj,
                np.mean(epoch_stats["test_loss"]),
                test_truth_obj,
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

        # Early stop if not improving
        earlyStopper(
            np.mean(epoch_stats["valid_loss"]), cbf_qp_net, stats, dir_dict["save_dir"]
        )

        # Log all statistics
        if not earlyStopper.counting_stop:
            if args["save_all_stats"]:
                if epoch == 0:
                    for key in epoch_stats.keys():
                        stats[key] = np.expand_dims(np.array(epoch_stats[key]), axis=0)
                else:
                    for key in epoch_stats.keys():
                        stats[key] = np.concatenate(
                            (
                                stats[key],
                                np.expand_dims(np.array(epoch_stats[key]), axis=0),
                            )
                        )
            else:
                stats = epoch_stats

        if earlyStopper.early_stop:
            utils.printInBoldRed("\nEarlyStopping: stop training!")
            with open(os.path.join(dir_dict["save_dir"], "stats.dict"), "rb") as f:
                final_data = pickle.load(f)
            utils.printInBoldGreen(
                "normalized train loss {:.4f}, valid loss {:.4f}, test loss {:.4f}".format(
                    train_truth_obj / np.mean(final_data["train_loss"][-1]),
                    valid_truth_obj / np.mean(final_data["valid_loss"][-1]),
                    test_truth_obj / np.mean(final_data["test_loss"][-1]),
                )
            )
            break

    if board:
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


def compute_truth_obj(dataset):
    """Compute the truth objective value/loss from the dataset"""
    sum = 0.0
    for i in range(len(dataset)):
        obj_val = dataset[i][2]
        sum += obj_val
    return sum / len(dataset)


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
def infer_net(model, data, args, dir_dict=None):
    "Intuitvely evaluate random test data by inference"

    dataset = TensorDataset(data.X, data.Y, data.obj_val)
    test_dataset = torch.utils.data.Subset(
        dataset,
        range(
            data.train_num + data.valid_num,
            data.train_num + data.valid_num + data.test_num,
        ),
    )

    model.load_state_dict(torch.load(dir_dict["infer_dir"]))
    model.eval()

    total_time = 0.0

    num = len(test_dataset)
    for i in range(128):
        idx = np.random.randint(0, num)
        X, Y, obj_val = test_dataset[idx]
        X = X.unsqueeze(0)
        start_time = time.time()
        Ynn = model(X).squeeze().numpy()
        total_time += time.time() - start_time
        Xo = X.squeeze().numpy()
        print(f"{Xo   = }")
        Yopt = Y.numpy()
        utils.printInBoldGreen(f"{Yopt = }\n{Ynn  = }")
        print("--")

    infer_time = total_time / 128
    print(f"{infer_time=}")


if __name__ == "__main__":
    main()
    print()
