import numpy as np
import math
import torch

A = np.array([True, True, False])
mytrue = np.alltrue(A)
# print(mytrue)

tens = torch.Tensor([[[0.3333], [0.3333], [0.3333]]])
print(tens[1, 0])
