import numpy as np
import sympy
from Diagonalization.states.multi_electrons import ChooseState

def D2_J(shift_orbs=0):
    # 0 1      2 3     4 5   6 7    8 9
    # d3z2-r2 dx2-y2   dyz   dxz    dxy
    up_orbs = [
        {6+shift_orbs,         8+shift_orbs},
        {4+shift_orbs,         8+shift_orbs},
        {7+shift_orbs,         9+shift_orbs},
        {5+shift_orbs,         9+shift_orbs}
    ]
    up_weight = [
        -1j/2,
        -1/2,
        1j/2,
        -1/2
    ]

    s2 = np.sqrt(2)
    s3 = np.sqrt(3)
    # 0 1      2 3     4 5   6 7    8 9
    # d3z2-r2 dx2-y2   dyz   dxz    dxy
    dn_orbs = [
        {6+shift_orbs,         8+shift_orbs},
        {4+shift_orbs,         8+shift_orbs},
        {7+shift_orbs,         9+shift_orbs},
        {5+shift_orbs,         9+shift_orbs},
        {4+shift_orbs,         7+shift_orbs},
        {5+shift_orbs,         6+shift_orbs}
    ]
    dn_weight = [
        1j/(2*s3),
        -1/(2*s3),
        -1j/(2*s3),
        -1/(2*s3),
        1j/(s3),
        1j/(s3)
    ]
    return up_orbs, up_weight, dn_orbs, dn_weight

def OneSiteD2J(basis, num_orbs=10):
    up_orbs1, up_weight1, dn_orbs1, dn_weight1 = D2_J(0)

    state1 = np.zeros((len(basis), 1), dtype=np.complex128)
    state2 = np.zeros((len(basis), 1), dtype=np.complex128)

    # up
    for i in range(0, 4):
        state1 += up_weight1[i] * ChooseState(basis, up_orbs1[i])

    # dn
    for i in range(0, 6):
        state2 += dn_weight1[i]  * ChooseState(basis, dn_orbs1[i])

    return np.hstack((state1, state2))

def TwoSiteD2J(basis, num_orbs=10):
    up_orbs1, up_weight1, dn_orbs1, dn_weight1 = D2_J(0)
    up_orbs2, up_weight2, dn_orbs2, dn_weight2 = D2_J(10)

    state1 = np.zeros((len(basis), 1), dtype=np.complex128)
    state2 = np.zeros((len(basis), 1), dtype=np.complex128)
    state3 = np.zeros((len(basis), 1), dtype=np.complex128)
    state4 = np.zeros((len(basis), 1), dtype=np.complex128)

    # up-dn
    for i in range(0, 4):
        for j in range(0, 6):
            state1 += up_weight1[i] * dn_weight2[j] * ChooseState(basis, up_orbs1[i].union(dn_orbs2[j]))

    # dn-up
    for i in range(0, 6):
        for j in range(0, 4):
            state2 += dn_weight1[i] * up_weight2[j] * ChooseState(basis, dn_orbs1[i].union(up_orbs2[j]))

    # up-up
    for i in range(0, 4):
        for j in range(0, 4):
            state3 += up_weight1[i] * up_weight2[j] * ChooseState(basis, up_orbs1[i].union(up_orbs2[j]))

    # dn-dn
    for i in range(0, 6):
        for j in range(0, 6):
            state4 += dn_weight1[i] * dn_weight2[j] * ChooseState(basis, dn_orbs1[i].union(dn_orbs2[j]))

    return np.hstack((state3, state1, state2, state4))