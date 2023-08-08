""" To store all examples for the online version
"""

import torch


class RppExample:
    def __init__(self, example=0):
        self.example = example

        if self.example == 0:  # A 2D polygon embedded in 3D
            self.name = "2D polygon embedded in 3D"
            self.xv_dim = 3  # embedded space
            self.xc_dim = 1
            self.y_dim = 3  # ambient space
            self.num_cstr = [0, 0, 0, 0, 0]  # linear ineq, linear eq, qcs, socs, lmis
            self.constraintInputMap = self.exampleMap0

        if self.example == 1:  # A polygon embedded in 3D with an sphere
            self.name = "A polygon embedded in 3D with an sphere"
            self.xv_dim = 3
            self.xc_dim = 1
            self.y_dim = 3
            self.num_cstr = [0, 0, 1, 0, 0]  # linear ineq, linear eq, qcs, socs, lmis
            self.constraintInputMap = self.exampleMap1

        if self.example == 2:  # Just a sphere with varying center and radius
            self.name = "sphere with xc1-3 center and xc0 radius"
            self.xv_dim = 3
            self.xc_dim = 4
            self.y_dim = 3
            self.num_cstr = [0, 0, 1, 0, 0]  # linear ineq, linear eq, qcs, socs, lmis
            self.constraintInputMap = self.exampleMap2

        elif self.example == 3:  # Just a paraboloid
            self.name = "paraboloid with q = q0 - xc"
            self.xv_dim = 3
            self.xc_dim = 3
            self.y_dim = 3
            self.num_cstr = [0, 0, 1, 0, 0]  # linear ineq, linear eq, qcs, socs, lmis
            self.constraintInputMap = self.exampleMap3

        elif self.example == 4:  # A 2d polyhedron
            self.name = "2d polyhedron with b = b0 * xc"
            self.xv_dim = 2
            self.xc_dim = 1
            self.y_dim = 2
            self.num_cstr = [0, 0, 0, 0, 0]  # linear ineq, linear eq, qcs, socs, lmis
            self.constraintInputMap = self.exampleMap4

        elif self.example == 5:  # A 2d polyhedron with a circle
            self.name = (
                "2d polyhedron b = b0 * xc0, with a circle xc2-3 center and xc1 radius"
            )
            self.xv_dim = 2
            self.xc_dim = 4
            self.y_dim = 2
            self.num_cstr = [0, 0, 1, 0, 0]  # linear ineq, linear eq, qcs, socs, lmis
            self.constraintInputMap = self.exampleMap5

        elif self.example == 6:  # The intersection between a cube and two planes
            self.name = "The intersection between a cube and two planes"
            self.xv_dim = 3
            self.xc_dim = 4
            self.y_dim = 3
            self.num_cstr = [0, 0, 0, 0, 0]  # linear ineq, linear eq, qcs, socs, lmis
            self.constraintInputMap = self.exampleMap6

        elif self.example == 7:  # Just a plane
            self.name = "Just a plane"
            self.xv_dim = 2
            self.xc_dim = 4
            self.y_dim = 3
            self.num_cstr = [0, 0, 0, 0, 0]  # linear ineq, linear eq, qcs, socs, lmis
            self.constraintInputMap = self.exampleMap7

        elif self.example == 8:
            # Unbounded 2d polyhedron. It has two vertices and two rays
            self.name = "unbounded 2d polyhedron b = b0 + xc"
            self.xv_dim = 2
            self.xc_dim = 3
            self.y_dim = 2
            self.num_cstr = [0, 0, 0, 0, 0]  # linear ineq, linear eq, qcs, socs, lmis
            self.constraintInputMap = self.exampleMap8

        elif self.example == 9:  # A paraboloid and a plane
            self.name = "a paraboloid and a plane"
            self.xv_dim = 2
            self.xc_dim = 3
            self.y_dim = 3
            self.num_cstr = [0, 0, 0, 0, 0]  # linear ineq, linear eq, qcs, socs, lmis
            self.constraintInputMap = self.exampleMap9

        elif self.example == 10:  # A paraboloid and a sphere
            self.name = "paraboloid with and a sphere with xc1-3 center and xc0 radius"
            self.xv_dim = 3
            self.xc_dim = 4
            self.y_dim = 3
            self.num_cstr = [0, 0, 2, 0, 0]  # linear ineq, linear eq, qcs, socs, lmis
            self.constraintInputMap = self.exampleMap10

        elif self.example == 11:  # A second-order cone
            self.name = "second-order cone with s = s0 - x"
            self.xv_dim = 3
            self.xc_dim = 3
            self.y_dim = 3
            self.num_cstr = [0, 0, 0, 1, 0]  # linear ineq, linear eq, qcs, socs, lmis
            self.constraintInputMap = self.exampleMap11

        elif self.example == 14:  # # Polyhedron and Ellipsoid
            self.name = "polyhedron b = b0 + xc with and ellipsoid"
            self.xv_dim = 3
            self.xc_dim = 2
            self.y_dim = 3
            self.num_cstr = [0, 0, 1, 0, 0]  # linear ineq, linear eq, qcs, socs, lmis
            self.constraintInputMap = self.exampleMap14

        elif self.example == 12:  # The PSD cone in 3D
            self.name = "PSD cone in 3D"
            self.xv_dim = 3
            self.xc_dim = 1
            self.y_dim = 3
            self.num_cstr = [0, 0, 0, 0, 1]  # linear ineq, linear eq, qcs, socs, lmis
            self.constraintInputMap = self.exampleMap12

        elif self.example == 13:  # Polyhedron, Ellipsoid, SOC, PSD
            self.name = "Polyhedron, Ellipsoid, SOC, PSD"
            self.xv_dim = 3
            self.xc_dim = 3
            self.y_dim = 3
            self.num_cstr = [0, 0, 1, 1, 1]  # linear ineq, linear eq, qcs, socs, lmis
            self.constraintInputMap = self.exampleMap13

        elif self.example == 20:  # High-dimensional sphere
            self.name = "High-dimensional sphere"
            self.xv_dim = 100
            self.xc_dim = 3
            self.y_dim = 100
            self.num_cstr = [0, 0, 1, 0, 0]
            self.constraintInputMap = self.exampleMap20

    def exampleMap0(self, x):  # A 2D polygon embedded in 3D
        A1, b1 = getCube()
        A2 = torch.tensor([[1.0, 1.0, 1.0]])
        b2 = torch.tensor([[1.0]])
        P, P_sqrt, q, r = getNoneQuadraticConstraints()
        M, s, c, d = getNoneSocConstraints()
        F = getNoneLmiConstraints()
        return A1, b1, A2, b2, P, P_sqrt, q, r, M, s, c, d, F

    def exampleMap1(self, x):  # A polygon embedded in 3D with an sphere
        A1, b1 = getCube()
        A2 = torch.tensor([[1.0, 1.0, 1.0]])
        b2 = torch.tensor([[1.0]])
        P, P_sqrt, q, r = getSphereConstraint(0.8, torch.zeros((3, 1)))
        M, s, c, d = getNoneSocConstraints()
        F = getNoneLmiConstraints()
        return A1, b1, A2, b2, P, P_sqrt, q, r, M, s, c, d, F

    def exampleMap2(self, x):  # Just a sphere
        A1, b1, A2, b2 = getNoneLinearConstraints()

        temp1 = torch.tensor([[1.0, 0, 0, 0]])
        temp2 = torch.tensor([[0, 1, 0, 0], [0, 0.0, 1, 0], [0, 0, 1, 0]])
        r = temp1 @ x
        c = temp2 @ x
        P, P_sqrt, q, r = getSphereConstraint(r, c)

        M, s, c, d = getNoneSocConstraints()
        F = getNoneLmiConstraints()
        return A1, b1, A2, b2, P, P_sqrt, q, r, M, s, c, d, F

    def exampleMap3(self, x):  # Just a paraboloid
        A1, b1, A2, b2 = getNoneLinearConstraints()
        P, P_sqrt, q, r = getParaboloid3DConstraint(x)
        M, s, c, d = getNoneSocConstraints()
        F = getNoneLmiConstraints()
        return A1, b1, A2, b2, P, P_sqrt, q, r, M, s, c, d, F

    def exampleMap4(self, x):  # A 2d polyhedron
        A1 = torch.tensor([[-1, 0], [0, -1.0], [0, 1.0], [0.6, 0.9701]])
        b1 = torch.tensor([[0], [0], [1], [1.2127]]) * x
        A2, b2 = getEmpty(), getEmpty()
        P, P_sqrt, q, r = getNoneQuadraticConstraints()
        M, s, c, d = getNoneSocConstraints()
        F = getNoneLmiConstraints()
        return A1, b1, A2, b2, P, P_sqrt, q, r, M, s, c, d, F

    def exampleMap5(self, x):  # A 2d polyhedron with a circle
        A1 = torch.tensor([[-1, 0], [0, -1.0], [0, 1.0], [0.6, 0.9701]])
        b1 = torch.tensor([[0], [0], [1], [1.2127]])
        A2, b2 = getEmpty(), getEmpty()
        temp1 = torch.tensor([[0.0, 1, 0, 0]])
        temp2 = torch.tensor([[0, 0.0, 1, 0], [0, 0, 0, 1]])
        r = temp1 @ x * 0.5
        c = temp2 @ x * 0.1
        P, P_sqrt, q, r = getSphereConstraint(r, c)
        M, s, c, d = getNoneSocConstraints()
        F = getNoneLmiConstraints()
        return A1, b1, A2, b2, P, P_sqrt, q, r, M, s, c, d, F

    def exampleMap6(self, x):  # The intersection between a cube and two planes
        A1, b1 = getCube()
        A2 = torch.tensor([[1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]])
        b2 = torch.tensor([[1.0], [0.1]])
        P, P_sqrt, q, r = getSphereConstraint(r, c)
        M, s, c, d = getNoneSocConstraints()
        F = getNoneLmiConstraints()
        return A1, b1, A2, b2, P, P_sqrt, q, r, M, s, c, d, F

    def exampleMap7(self, x):  # Just a plane
        A1, b1 = getEmpty(), getEmpty()
        A2 = torch.tensor([[1.0, 1.0, 1.0]])
        b2 = torch.tensor([[1.0]])
        P, P_sqrt, q, r = getSphereConstraint(r, c)
        M, s, c, d = getNoneSocConstraints()
        F = getNoneLmiConstraints()
        return A1, b1, A2, b2, P, P_sqrt, q, r, M, s, c, d, F

    def exampleMap8(self, x):
        # Unbounded 2d polyhedron. It has two vertices and two rays
        A1 = torch.tensor([[0.0, -1.0], [2.0, -4.0], [-2.0, 1.0]])
        b1 = torch.tensor([[-2.0], [1.0], [-5.0]]) - x
        A2, b2 = getEmpty(), getEmpty()
        P, P_sqrt, q, r = getNoneQuadraticConstraints()
        M, s, c, d = getNoneSocConstraints()
        F = getNoneLmiConstraints()
        return A1, b1, A2, b2, P, P_sqrt, q, r, M, s, c, d, F

    def exampleMap9(self, x):  # A paraboloid and a plane
        A1, b1 = getEmpty(), getEmpty()
        A2 = torch.tensor([[1.0, 1.0, 3.0]])
        b2 = torch.tensor([[1.0]])
        P, P_sqrt, q, r = getParaboloid3DConstraint()
        M, s, c, d = getNoneSocConstraints()
        F = getNoneLmiConstraints()
        return A1, b1, A2, b2, P, P_sqrt, q, r, M, s, c, d, F

    def exampleMap10(self, x):  # A paraboloid and a sphere
        A1, b1, A2, b2 = getNoneLinearConstraints()
        P1, P1_sqrt, q1, r1 = getParaboloid3DConstraint()
        temp1 = torch.tensor([[1.0, 0, 0, 0]])
        temp2 = torch.tensor([[0, 1, 0, 0], [0, 0.0, 1, 0], [0, 0, 1, 0]])
        r = temp1 @ x
        c = temp2 @ x * 0.1
        P2, P2_sqrt, q2, r2 = getSphereConstraint(r, c)
        P = torch.cat((P1, P2), dim=0)
        P_sqrt = torch.cat((P1_sqrt, P2_sqrt), dim=0)
        q = torch.cat((q1, q2), dim=0)
        r = torch.cat((r1, r2), dim=0)
        M, s, c, d = getNoneSocConstraints()
        F = getNoneLmiConstraints()
        return A1, b1, A2, b2, P, P_sqrt, q, r, M, s, c, d, F

    def exampleMap11(self, x):  # A second-order cone
        A1, b1, A2, b2 = getNoneLinearConstraints()
        P, P_sqrt, q, r = getNoneQuadraticConstraints()
        M, s, c, d = getSOC3DConstraint()
        F = getNoneLmiConstraints()
        return A1, b1, A2, b2, P, P_sqrt, q, r, M, s, c, d, F

    def exampleMap14(self, x):  # Polyhedron and Ellipsoid
        A1 = torch.tensor([[-1.0, -1.0, -1.0], [-1.0, 2.0, 2.0]])
        b1 = torch.tensor([[-1.0], [1.0]]) + x
        A2, b2 = getEmpty(), getEmpty()
        E_ellipsoid = torch.tensor([[0.6, 0, 0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        P, P_sqrt, q, r = getEllipsoidConstraint(E_ellipsoid, torch.zeros((3, 1)))
        M, s, c, d = getNoneSocConstraints()
        F = getNoneLmiConstraints()
        return A1, b1, A2, b2, P, P_sqrt, q, r, M, s, c, d, F

    def exampleMap12(self, x=None):  # PSD cone
        A1, b1, A2, b2 = getNoneLinearConstraints()
        P, P_sqrt, q, r = getNoneQuadraticConstraints()
        M, s, c, d = getNoneSocConstraints()
        F = getPSDCone3DConstraint(x)
        return A1, b1, A2, b2, P, P_sqrt, q, r, M, s, c, d, F

    def exampleMap13(self, x):  # Polyhedron and Ellipsoid
        A1 = torch.tensor([[-1.0, -1.0, -1.0]])
        b1 = torch.tensor([[-1.0]])
        A2, b2 = getEmpty(), getEmpty()
        E_ellipsoid = torch.tensor([[0.1, 0, 0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        P, P_sqrt, q, r = getEllipsoidConstraint(E_ellipsoid, torch.zeros((3, 1)))
        M, s, c, d = getSOC3DConstraint()
        F = getPSDCone3DConstraint()
        return A1, b1, A2, b2, P, P_sqrt, q, r, M, s, c, d, F

    def exampleMap20(self, x):  # High-dimensional sphere
        A1, b1, A2, b2 = getNoneLinearConstraints()
        P, P_sqrt, q, r = getSphereConstraint(5, torch.zeros(self.xv_dim, 1))
        M, s, c, d = getNoneSocConstraints()
        F = getNoneLmiConstraints()
        return A1, b1, A2, b2, P, P_sqrt, q, r, M, s, c, d, F


def getEmpty():
    return torch.tensor([])


def getNoneLinearConstraints():
    return getEmpty(), getEmpty(), getEmpty(), getEmpty()


def getNoneQuadraticConstraints():
    return getEmpty(), getEmpty(), getEmpty(), getEmpty()


def getNoneSocConstraints():
    return getEmpty(), getEmpty(), getEmpty(), getEmpty()


def getNoneLmiConstraints():
    return getEmpty()


def getCube():
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
    return A1, b1


# Ellipsoid is defined as {x | (x-c)'E(x-c)<=1}
# Where E is a positive semidefinite matrix
def getEllipsoidConstraint(E, c):
    # Convert to (1/2)x'P_ix + q_i'x +r_i <=0
    P = 2 * E
    P_sqrt = torch.sqrt(P)
    q = -2 * E @ c
    r = c.transpose(-1, -2) @ E @ c - 1
    return P, P_sqrt, q, r


# Sphere of radius r centered around c
def getSphereConstraint(r, c):
    return getEllipsoidConstraint((1 / (r * r)) * torch.eye(c.shape[0]), c)


def getParaboloid3DConstraint(x=None):
    if x is None:
        P = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        P_sqrt = torch.sqrt(P)
        q = torch.tensor([[0.0], [0.0], [-1.0]])
        r = torch.tensor([[0.0]])
    else:
        P = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        P_sqrt = torch.sqrt(P)
        q = torch.tensor([[0.0], [0.0], [-1.0]]) - x
        r = torch.tensor([[0.0]])
    return P, P_sqrt, q, r


def getSOC3DConstraint(x=None):
    if x is None:
        M = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        s = torch.tensor([[0.0], [0.0], [0.0]])
        c = torch.tensor([[0.0], [0.0], [1.0]])
        d = torch.tensor([[0.0]])
    else:
        M = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        s = torch.tensor([[0.0], [0.0], [0.0]]) - x
        c = torch.tensor([[0.0], [0.0], [1.0]])
        d = torch.tensor([[0.0]])
    return M, s, c, d


def getPSDCone3DConstraint(x=None):
    # [x y;y z] >> 0

    if x is None:
        F0 = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        F1 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        F2 = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
        F3 = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    else:
        F0 = torch.tensor([[1.0, 0.0], [0.0, 0.0]]) * x
        F1 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        F2 = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
        F3 = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    F = torch.cat((F0, F1, F2, F3), dim=0)
    return F
