import numpy as np
from Diagonalization.states.multi_electrons import ChooseState


def J1_2(shift_orbs=0, num_orbs=3):
    up_orbs = [2+shift_orbs, 1+num_orbs+shift_orbs, 0+num_orbs+shift_orbs]
    up_weight = np.array([1/np.sqrt(3), 1j/np.sqrt(3), 1/np.sqrt(3)], dtype=np.complex128)
    dn_orbs = [2+num_orbs+shift_orbs, 1+shift_orbs, 0+shift_orbs]
    dn_weight = np.array([1/np.sqrt(3), 1j/np.sqrt(3), -1/np.sqrt(3)], dtype=np.complex128)
    return up_orbs, up_weight, dn_orbs, dn_weight


def TwoSiteJ(basis):
    up_orbs1, up_weight1, dn_orbs1, dn_weight1 = J1_2(0, 6)
    up_orbs2, up_weight2, dn_orbs2, dn_weight2 = J1_2(3, 6)

    # up-dn
    state1 = np.zeros((len(basis), 1), dtype=np.complex128)
    for i in range(0, 3):
        for j in range(0, 3):
            state1 += up_weight1[i] * dn_weight2[j] * ChooseState(basis, {up_orbs1[i], dn_orbs2[j]})

    # dn-up
    state2 = np.zeros((len(basis), 1), dtype=np.complex128)
    for i in range(0, 3):
        for j in range(0, 3):
            state2 += dn_weight1[i] * up_weight2[j] * ChooseState(basis, {dn_orbs1[i], up_orbs2[j]})

    # up-up
    state3 = np.zeros((len(basis), 1), dtype=np.complex128)
    for i in range(0, 3):
        for j in range(0, 3):
            state3 += up_weight1[i] * up_weight2[j] * ChooseState(basis, {up_orbs1[i], up_orbs2[j]})

    # dn-dn
    state4 = np.zeros((len(basis), 1), dtype=np.complex128)
    for i in range(0, 3):
        for j in range(0, 3):
            state4 += dn_weight1[i] * dn_weight2[j] * ChooseState(basis, {dn_orbs1[i], dn_orbs2[j]})

    return np.hstack((state3, state1, state2, state4))