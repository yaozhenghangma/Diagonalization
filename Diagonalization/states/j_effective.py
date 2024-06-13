import numpy as np
import sympy
from Diagonalization.states.multi_electrons import ChooseState


def J1_2(shift=0):
    up_orbs = [0+shift, 2+shift, 5+shift]
    up_weight = np.array([1/np.sqrt(3), -1j/np.sqrt(3), -1/np.sqrt(3)], dtype=np.complex128)
    dn_orbs = [1+shift, 3+shift, 4+shift]
    dn_weight = np.array([-1j/np.sqrt(3), 1/np.sqrt(3), -1j/np.sqrt(3)], dtype=np.complex128)
    return up_orbs, up_weight, dn_orbs, dn_weight


def TwoSiteJ(basis):
    up_orbs1, up_weight1, dn_orbs1, dn_weight1 = J1_2(0)
    up_orbs2, up_weight2, dn_orbs2, dn_weight2 = J1_2(6)


    state1 = np.zeros((len(basis), 1), dtype=np.complex128)
    state2 = np.zeros((len(basis), 1), dtype=np.complex128)
    state3 = np.zeros((len(basis), 1), dtype=np.complex128)
    state4 = np.zeros((len(basis), 1), dtype=np.complex128)

    # up-dn
    for i in range(0, 3):
        for j in range(0, 3):
            if up_orbs1[i] < dn_orbs2[j]:
                state1 += up_weight1[i] * dn_weight2[j] * ChooseState(basis, {up_orbs1[i], dn_orbs2[j]})
            else:
                state1 -= up_weight1[i] * dn_weight2[j] * ChooseState(basis, {up_orbs1[i], dn_orbs2[j]})

    # dn-up
    for i in range(0, 3):
        for j in range(0, 3):
            if dn_orbs1[i] < up_orbs2[j]:
                state2 += dn_weight1[i] * up_weight2[j] * ChooseState(basis, {dn_orbs1[i], up_orbs2[j]})
            else:
                state2 -= dn_weight1[i] * up_weight2[j] * ChooseState(basis, {dn_orbs1[i], up_orbs2[j]})

    # up-up
    for i in range(0, 3):
        for j in range(0, 3):
            if up_orbs1[i] < up_orbs2[j]:
                state3 += up_weight1[i] * up_weight2[j] * ChooseState(basis, {up_orbs1[i], up_orbs2[j]})
            else:
                state3 -= up_weight1[i] * up_weight2[j] * ChooseState(basis, {up_orbs1[i], up_orbs2[j]})

    # dn-dn
    for i in range(0, 3):
        for j in range(0, 3):
            if dn_orbs1[i] < dn_orbs2[j]:
                state4 += dn_weight1[i] * dn_weight2[j] * ChooseState(basis, {dn_orbs1[i], dn_orbs2[j]})
            else:
                state4 -= dn_weight1[i] * dn_weight2[j] * ChooseState(basis, {dn_orbs1[i], dn_orbs2[j]})

    return np.hstack((state3, state1, state2, state4))