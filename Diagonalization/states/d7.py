import numpy as np
import sympy
from Diagonalization.states.multi_electrons import ChooseState

def D7_J1_2(shift_orbs=0, num_orbs=5):
    up_orbs = [
        {0+shift_orbs,         1+shift_orbs,          2+shift_orbs},
        {0+shift_orbs,         1+shift_orbs,          3+shift_orbs},
        {0+shift_orbs,         1+shift_orbs,          4+shift_orbs+num_orbs},
        {0+shift_orbs,         4+shift_orbs,          1+shift_orbs+num_orbs},
        {0+shift_orbs,         1+shift_orbs+num_orbs, 2+shift_orbs+num_orbs},
        {0+shift_orbs,         1+shift_orbs+num_orbs, 3+shift_orbs+num_orbs},
        {1+shift_orbs,         4+shift_orbs,          0+shift_orbs+num_orbs},
        {1+shift_orbs,         0+shift_orbs+num_orbs, 2+shift_orbs+num_orbs},
        {1+shift_orbs,         0+shift_orbs+num_orbs, 3+shift_orbs+num_orbs},
        {2+shift_orbs,         0+shift_orbs+num_orbs, 1+shift_orbs+num_orbs},
        {3+shift_orbs,         0+shift_orbs+num_orbs, 1+shift_orbs+num_orbs}
    ]
    up_weight = [
        1/2,
        -1j/2,
        1/3,
        1/3,
        1/6,
        1j/6,
        1/3,
        1/6,
        1j/6,
        1/6,
        1j/6
    ]

    dn_orbs = [
        {0+shift_orbs,         1+shift_orbs,          2+shift_orbs+num_orbs},
        {0+shift_orbs,         1+shift_orbs,          3+shift_orbs+num_orbs},
        {0+shift_orbs,         2+shift_orbs,          1+shift_orbs+num_orbs},
        {0+shift_orbs,         3+shift_orbs,          1+shift_orbs+num_orbs},
        {0+shift_orbs,         1+shift_orbs+num_orbs, 4+shift_orbs+num_orbs},
        {1+shift_orbs,         2+shift_orbs,          0+shift_orbs+num_orbs},
        {1+shift_orbs,         3+shift_orbs,          0+shift_orbs+num_orbs},
        {1+shift_orbs,         0+shift_orbs+num_orbs, 4+shift_orbs+num_orbs},
        {4+shift_orbs,         0+shift_orbs+num_orbs, 1+shift_orbs+num_orbs},
        {0+shift_orbs+num_orbs,1+shift_orbs+num_orbs, 2+shift_orbs+num_orbs},
        {0+shift_orbs+num_orbs,1+shift_orbs+num_orbs, 3+shift_orbs+num_orbs}
    ]
    dn_weight = [
        1/6,
        -1j/6,
        1/6,
        -1j/6,
        -1/3,
        1/6,
        -1j/6,
        -1/3,
        -1/3,
        1/2,
        1j/2
    ]
    return up_orbs, up_weight, dn_orbs, dn_weight

def TwoSiteD7J(basis, num_orbs=10):
    up_orbs1, up_weight1, dn_orbs1, dn_weight1 = D7_J1_2(0, num_orbs)
    up_orbs2, up_weight2, dn_orbs2, dn_weight2 = D7_J1_2(5, num_orbs)

    state1 = np.zeros((len(basis), 1), dtype=np.complex128)
    state2 = np.zeros((len(basis), 1), dtype=np.complex128)
    state3 = np.zeros((len(basis), 1), dtype=np.complex128)
    state4 = np.zeros((len(basis), 1), dtype=np.complex128)

    # up-dn
    for i in range(0, 11):
        for j in range(0, 11):
            state1 += up_weight1[i] * dn_weight2[j] * ChooseState(basis, up_orbs1[i].union(dn_orbs2[j]))

    # dn-up
    for i in range(0, 11):
        for j in range(0, 11):
            state2 += dn_weight1[i] * up_weight2[j] * ChooseState(basis, dn_orbs1[i].union(up_orbs2[j]))

    # up-up
    for i in range(0, 11):
        for j in range(0, 11):
            state3 += up_weight1[i] * up_weight2[j] * ChooseState(basis, up_orbs1[i].union(up_orbs2[j]))

    # dn-dn
    for i in range(0, 11):
        for j in range(0, 11):
            state4 += dn_weight1[i] * dn_weight2[j] * ChooseState(basis, dn_orbs1[i].union(dn_orbs2[j]))

    return np.hstack((state3, state1, state2, state4))