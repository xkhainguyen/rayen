import numpy as np
import math
import torch
import torch.nn as nn
from scipy.linalg import null_space
import utils_examples

import fixpath  # Following this example: https://github.com/tartley/colorama/blob/master/demos/demo01.py

# from rayen import utils, constraint_module, constraints_torch


# Set the default device to GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_tensor_type(
    torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor
)
torch.set_default_dtype(torch.float64)


a = np.array([[[1, 2, 3], [4.0, 5, 6]], [[1, 2, 3], [4.0, 5, 6]]])
print(a.transpose(-1, -2))
