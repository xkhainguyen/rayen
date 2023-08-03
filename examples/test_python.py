import numpy as np
import math
import torch
import torch.nn as nn
from scipy.linalg import null_space
import utils_examples

import fixpath  # Following this example: https://github.com/tartley/colorama/blob/master/demos/demo01.py
from rayen import utils, constraint_module, constraints_torch


torch.set_default_dtype(torch.float64)
# Set the default device to GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_tensor_type(
    torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor
)
# # Create a batch of matrices (3x3 matrices, batch size 2)
# batch_size = 2
# matrix_size = (batch_size, 3, 3)
# batch_A = torch.randn(matrix_size)
# batch_A = torch.tensor([[[1.0, 1.0, 1.0]], [[-1.0, -1.0, -1.0]]])
# A = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
# # Compute the SVD for each matrix in the batch
# U, S, V = torch.svd(batch_A[0], some=False, compute_uv=True)
# S = S.unsqueeze(-1)
# V = V.unsqueeze(0)
# print(U)
# print(S)
# print(V)
# # Set a tolerance value for identifying singular values close to zero
# tolerance = torch.full((S.shape), torch.finfo(S.dtype).eps)

# # Find the indices of singular values close to zero (indicating null space)
# null_space_mask = (S < tolerance).squeeze(-1)
# print(null_space_mask)
# # Compute the null space bases for each matrix in the batch using broadcasting
# null_space_bases = (V.transpose(-1, -2))[null_space_mask]

# # Transpose the null space bases to the correct shape
# null_space_bases = null_space_bases.transpose(-1, -2)
# print(A)
# null_space_bases1 = null_space(A)
# # Print the null space basis vectors
# print("Null Space Basis Vectors:")
# print(null_space_bases)
# print("\nNumpy:")
# print(null_space_bases1)
# print("\nNew:")


# def my_nullspace(At, rcond=None):
#     ut, st, vht = torch.Tensor.svd(At, some=False, compute_uv=True)
#     vht = vht.transpose(-1, -2)
#     st = st.unsqueeze(-1)
#     # print(ut)
#     print(st)
#     # print(vht)
#     Mt, Nt = ut.shape[1], vht.shape[2]
#     # print(Mt, Nt)
#     if rcond is None:
#         rcondt = torch.finfo(st.dtype).eps * max(Mt, Nt)
#     # print(rcondt)
#     # print(torch.max(st, 1)[0])
#     tolt = (torch.max(st, 1)[0] * rcondt).unsqueeze(-1)
#     print(tolt)
#     print(st < tolt)
#     numt = torch.sum(st > tolt, dim=1, dtype=int)
#     print(numt)
#     print(vht)
#     numt = torch.tensor([[1], [2]])
#     # print(numt)
#     nullspace = vht[:, numt:, :].transpose(-1, -2).conj()
#     # nullspace.backward(torch.ones_like(nullspace),retain_graph=True)
#     return nullspace


# def my_nullspace(At, rcond=None):
#     ut, st, vht = torch.Tensor.svd(At, some=False, compute_uv=True)
#     vht = vht.T
#     print(ut)
#     print(st)
#     print(vht)
#     Mt, Nt = ut.shape[0], vht.shape[1]
#     print(Mt, Nt)
#     if rcond is None:
#         rcondt = torch.finfo(st.dtype).eps * max(Mt, Nt)
#     print(rcondt)
#     tolt = torch.max(st) * rcondt
#     print(tolt)
#     print(st > tolt)
#     numt = torch.sum(st > tolt, dtype=int)
#     print(numt)
#     print(vht)
#     nullspace = vht[numt:, :].T.conj()
#     # nullspace.backward(torch.ones_like(nullspace),retain_graph=True)
#     return nullspace


# print(my_nullspace(batch_A))


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

# P_sqrt = [A[:, i : i + 2, :] for i in range(2)]
# print(P_sqrt)

A1 = torch.eye(3)
P1_sqrt = torch.sqrt(A1)
A2 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
P2_sqrt = torch.sqrt(A2)
P = torch.cat((P1_sqrt, P2_sqrt), dim=0)
print(P)
