import numpy as np
import sympy
from Diagonalization.states.multi_electrons import ChooseState


def J1_2(shift_orbs=0, num_orbs=3, symbolic=False):
    if symbolic:
        up_orbs = [2 + shift_orbs, 1 + num_orbs + shift_orbs, 0 + num_orbs + shift_orbs]
        up_weight = sympy.Matrix([1 / sympy.sqrt(3), 1j / sympy.sqrt(3), 1 / sympy.sqrt(3)])
        dn_orbs = [2 + num_orbs + shift_orbs, 1 + shift_orbs, 0 + shift_orbs]
        dn_weight = sympy.Matrix([1 / sympy.sqrt(3), 1j / sympy.sqrt(3), -1 / sympy.sqrt(3)])
        return up_orbs, up_weight, dn_orbs, dn_weight
    else:
        up_orbs = [2+shift_orbs, 1+num_orbs+shift_orbs, 0+num_orbs+shift_orbs]
        up_weight = np.array([1/np.sqrt(3), 1j/np.sqrt(3), 1/np.sqrt(3)], dtype=np.complex128)
        dn_orbs = [2+num_orbs+shift_orbs, 1+shift_orbs, 0+shift_orbs]
        dn_weight = np.array([1/np.sqrt(3), 1j/np.sqrt(3), -1/np.sqrt(3)], dtype=np.complex128)
        return up_orbs, up_weight, dn_orbs, dn_weight


def TwoSiteJ(basis, symbolic=False):
    up_orbs1, up_weight1, dn_orbs1, dn_weight1 = J1_2(0, 6, symbolic)
    up_orbs2, up_weight2, dn_orbs2, dn_weight2 = J1_2(3, 6, symbolic)

    if symbolic:
        state1 = sympy.Matrix.zeros(len(basis), 1)
        state2 = sympy.Matrix.zeros(len(basis), 1)
        state3 = sympy.Matrix.zeros(len(basis), 1)
        state4 = sympy.Matrix.zeros(len(basis), 1)
    else:
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