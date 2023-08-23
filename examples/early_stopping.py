# Code taken from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
# Some minor modifications by jtorde@mit.edu

import numpy as np
import pickle
import torch
import uuid
import os
import sys
from os.path import normpath, dirname, join

sys.path.insert(0, normpath(join(dirname(__file__), "../..")))


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.counting_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss, model, stats, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, stats, path)
        elif score < self.best_score * (1 + self.delta * np.sign(score)):
            if not self.counting_stop:
                self.save_checkpoint(val_loss, model, stats, path)
            self.counting_stop = True
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}, best valid loss {-self.best_score:.4f}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                self.counting_stop = True
        else:
            self.counting_stop = False
            self.best_score = score
            self.save_checkpoint(val_loss, model, stats, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, stats, save_dir):
        """Saves model when validation loss decrease."""
        self.val_loss_min = val_loss

        # if self.verbose:
        #     # self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #     self.trace_func(
        #         f"Best validation loss so far: {self.val_loss_min:.6f}.  Saving model ..."
        #     )
        # torch.save(model.state_dict(), self.path)
        with open(os.path.join(save_dir, "stats.dict"), "wb") as f:
            pickle.dump(stats, f)
        with open(os.path.join(save_dir, "model.dict"), "wb") as f:
            torch.save(model.state_dict(), f)

    # Added by jtorde
    # def load_best_model(self, model):
    #     model.load_state_dict(torch.load(self.path))

    # def __del__(self):
    #     os.system("rm " + self.path)

    # body of destructor
