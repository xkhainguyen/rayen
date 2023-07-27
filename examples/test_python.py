import numpy as np
import math
import torch

A = np.array([True, True, False])
mytrue = np.alltrue(A)
# print(mytrue)

tens = torch.Tensor([[[0.3333], [0.3333], [0.3333]]])
# print(tens[1, 0])


def customConstraintMap(x):
    # Modify the following
    A1 = torch.tensor(
        [
            [1.0, 0, 0],
            [0, 1.0, 0],
            [0, 0, 1.0],
            [-1.0, 0, 0],
            [0, -1.0, 0],
            [0, 0, -1.0],
        ]
    )
    b1 = torch.tensor([[1.0], [1.0], [1.0], [0], [0], [0]])
    A2 = torch.tensor([[1.0, 1.0, 1.0]])
    b2 = x[0, 0:1].unsqueeze(dim=1)
    return A1, b1, A2, b2


batch_size, feature_size = 2, 1
examples = torch.randn(batch_size, feature_size, 1)
A1, b1, A2, b2 = torch.vmap(customConstraintMap)(examples)
# x = torch.randn(2)
# A1, b1, A2, b2 = customConstraintMap(x)
# print(examples)
# print(A1)
# print(b1.size())
# print(A2)
# print(b2)
# print(b2.shape)

A = A1
b = b1

A = torch.cat((A, A2, -A2), axis=1)
b = torch.cat((b, b2, -b2), axis=1)

print(A)
print(b)
