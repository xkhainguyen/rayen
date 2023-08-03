""" To store all examples for the online version
"""

import torch


# Just a box with changing width
def constraintInputMap(x):
    # x is a rank-2 tensor
    # outputs are rank-2 tensors
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
    b1 = torch.tensor([[1.0], [1.0], [1.0], [0], [0], [0]]) @ x
    A2 = torch.tensor([])
    b2 = torch.tensor([])
    # A2 = torch.tensor([[1.0, 1.0, 1.0]])
    # b2 = x[0, 0:1].unsqueeze(dim=1)
    return A1, b1, A2, b2


# Two circles with changing centers
def constraintInputMap(x):
    # x is a rank-2 tensor
    # outputs are rank-2 tensors

    # Linear constraints
    A1 = torch.tensor([])
    b1 = torch.tensor([])
    A2 = torch.tensor([])
    b2 = torch.tensor([])

    # Quadratic constraints
    r1 = 2.0
    c1 = x
    E1 = (1 / (r1 * r1)) * torch.eye(2)
    P1 = 2 * E1
    q1 = -2 * E1 @ c1
    r1 = c1.transpose(-1, -2) @ E1 @ c1 - 1

    r2 = 2.0
    c2 = x * 0.9
    E2 = (1 / (r2 * r2)) * torch.eye(2)
    P2 = 2 * E2
    q2 = -2 * E2 @ c2
    r2 = c2.transpose(-1, -2) @ E2 @ c2 - 1

    P = torch.cat((P1, P2), dim=0)
    q = torch.cat((q1, q2), dim=0)
    r = torch.cat((r1, r2), dim=0)

    return A1, b1, A2, b2, P, q, r


# Polyhedron and Ellipsoid
def constraintInputMap(x):
    # x is a rank-2 tensor
    # outputs are rank-2 tensors

    # Linear constraints
    A1 = torch.tensor([[-1.0, -1.0, -1.0], [-1.0, 2.0, 2.0]])
    b1 = torch.tensor([[-1.0], [1.0]])
    A2 = torch.tensor([])
    b2 = torch.tensor([])

    # Quadratic constraints
    r1 = 1.0
    c1 = torch.zeros(3, 1) + 0.1 * x
    # print(f"x = {x}")
    E1 = torch.tensor([[0.1, 0, 0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    P1 = 2 * E1
    q1 = -2 * E1 @ c1
    r1 = c1.transpose(-1, -2) @ E1 @ c1 - 1

    P = P1
    P_sqrt = torch.sqrt(P)
    q = q1
    r = r1

    return A1, b1, A2, b2, P, P_sqrt, q, r


# Just a paraboloid
def constraintInputMap(x):
    # x is a rank-2 tensor
    # outputs are rank-2 tensors

    # Linear constraints
    A1 = torch.tensor([])
    b1 = torch.tensor([])
    A2 = torch.tensor([])
    b2 = torch.tensor([])

    # Quadratic constraints
    P = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    P_sqrt = torch.sqrt(P)
    q = torch.tensor([[0.0], [0.0], [-1.0]])
    r = torch.tensor([[0.0]]) + x

    return A1, b1, A2, b2, P, P_sqrt, q, r


# A paraboloid and a shpere
def constraintInputMap(x):
    # x is a rank-2 tensor
    # outputs are rank-2 tensors

    # Linear constraints
    A1 = torch.tensor([])
    b1 = torch.tensor([])
    A2 = torch.tensor([])
    b2 = torch.tensor([])

    # Quadratic constraints
    r1 = 2.0
    c1 = torch.zeros(3, 1) + 0.1 * x
    # print(f"x = {x}")
    E1 = (1 / (r1 * r1)) * torch.eye(3)
    P1 = 2 * E1
    P1_sqrt = torch.sqrt(P1)
    q1 = -2 * E1 @ c1
    r1 = c1.transpose(-1, -2) @ E1 @ c1 - 1

    P2 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    P2_sqrt = torch.sqrt(P2)
    q2 = torch.tensor([[0.0], [0.0], [-1.0]])
    r2 = torch.tensor([[0.0]])

    P = torch.cat((P1, P2), dim=0)
    P_sqrt = torch.cat((P1_sqrt, P2_sqrt), dim=0)
    q = torch.cat((q1, q2), dim=0)
    r = torch.cat((r1, r2), dim=0)

    return A1, b1, A2, b2, P, P_sqrt, q, r
