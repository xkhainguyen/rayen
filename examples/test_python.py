import numpy as np
import math
import torch
from scipy.linalg import null_space

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


# A = torch.tensor(
#     [[0.81649658, 0.0], [-0.40824829, -0.70710678], [-0.40824829, 0.70710678]]
# )
# A_batch = A.unsqueeze(0).repeat(5, 1, 1)
# print(A_batch)

batch_size = 2
n = 3
NA_E = torch.eye(n).unsqueeze(0).repeat(batch_size, 1, 1)
print(NA_E)
yp = torch.zeros(batch_size, n, 1)
print(yp)
A_p = torch.zeros(batch_size, 1, n)  # 0z<=1
print(A_p)
b_p = torch.ones(batch_size, 1, 1)
print(b_p)
A_E = torch.zeros(batch_size, 1, n)
print(A_E)
# 0y=0
b_E = torch.zeros(batch_size, 1, 1)
print(b_E)
A_I = torch.zeros(batch_size, 1, n)
print(A_I)
# 0y<=1
b_I = torch.ones(batch_size, 1, 1)
print(b_I)
