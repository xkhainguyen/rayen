import numpy as np
import math
import torch
import torch.nn as nn
from scipy.linalg import null_space
import utils_examples

import fixpath  # Following this example: https://github.com/tartley/colorama/blob/master/demos/demo01.py

# from rayen import utils, constraint_module, constraints_torch


torch.set_default_dtype(torch.float64)
# Set the default device to GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_tensor_type(
    torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor
)

batch_A = torch.tensor([[[1.0, 1.0, 1.0]], [[-1.0, -1.0, -1.0]]])
A = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])


def nullspace(At, tolt=0.0):
    ut, st, vht = torch.Tensor.svd(At, some=False)
    vht = vht.transpose(-1, -2)
    st = st.unsqueeze(1)
    print(f"ut={ut}")
    print(f"st={st}")
    print(f"vht={vht}")
    Mt, Nt = ut.shape[1], vht.shape[2]
    print(st > tolt)
    numt = torch.sum(st > tolt, dtype=int, dim=1).unsqueeze(1)
    print(f"numt={numt}")
    print(f"vht={vht}")
    nullspace = vht[:, numt:, :].transpose(-1, -2).conj()
    # nullspace.backward(torch.ones_like(nullspace),retain_graph=True)
    return nullspace


def nullspace2(A, tol=1e-12):
    U, S, V = torch.svd(A, some=False)
    if S.min() >= tol:
        null_start = len(S)
    else:
        null_start = int(len(S) - torch.sum(S < tol))

    V_null = V[:, null_start:]
    return V_null


print(nullspace(batch_A))
# print(nullSpaceBatch(batch_A))

B = torch.tensor(
    [
        [
            [-0.5774, -0.5774, -0.5774],
            [-0.5774, 0.7887, -0.2113],
            [-0.5774, -0.2113, 0.7887],
        ],
        [
            [-0.5774, -0.5774, -0.5774],
            [-0.5774, 0.7887, -0.2113],
            [-0.5774, -0.2113, 0.7887],
        ],
    ]
)

indexes = torch.tensor([[[0]], [[1]]])
# This means taking rows 0, 1, 2 from B[0] and taking rows 1, 2 from B[1]
# How to do it in batch?

C = torch.zeros(B.shape)

masks = torch.tensor([[True, True, False], [True, False, False]])

# A = torch.tensor([[[1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0]]])
# b = torch.tensor([[[1.0]], [[1.0]]])
# A1 = torch.tensor([[1.0, 1.0, 1.0]])
# b1 = torch.tensor([[1.0]])
# print(A)
# print(b)
# print(A.device)
# print(np.linalg.pinv(A[0].cpu()) @ b[0].detach().cpu().numpy())
# # print(torch.pinverse(A, driver="gesvd"))
# # s = torch.linalg.svdvals(A[0])
# # print(s)
# yp = torch.linalg.lstsq(A1, b1).solution
# print(yp)


# def pinv(A):
#     """
#     Return the pseudoinverse of A using the QR decomposition.
#     """
#     Q, R = torch.linalg.qr(A)
#     return R.pinverse() @ (Q.transpose(-1, -2))


# print(pinv(A))
# Create a 2x3 matrix
# A = torch.tensor(
#     [[[4, 0], [0, 4], [2, 0.0], [0, 2]], [[1, 0], [0, 1.0], [3, 0], [0, 3]]]
# )

# # Compute the pseudo-inverse of A
# A_pinv = torch.pinverse(A)

# print("Original matrix A:")
# print(A)

# print("\nPseudo-inverse of A:")
# print(A_pinv)

# A = torch.tensor([[1, 1, 1.0], [-1, -1, -1.0]])


# def my_nullspace(At, rcond=None):
#     ut, st, vht = torch.Tensor.svd(At, some=False, compute_uv=True)
#     vht = vht.T
#     Mt, Nt = ut.shape[0], vht.shape[1]
#     if rcond is None:
#         rcondt = torch.finfo(st.dtype).eps * max(Mt, Nt)
#     tolt = torch.max(st) * rcondt
#     numt = torch.sum(st > tolt, dtype=int)
#     nullspace = vht[numt:, :].T.cpu().conj()
#     # nullspace.backward(torch.ones_like(nullspace),retain_graph=True)
#     return nullspace


# out = my_nullspace(A)
# print(out)
