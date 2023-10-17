import numpy as np
import sympy


def ChooseState(basis, orbs, symbolic=False):
    for i in range(0, len(basis)):
        if orbs == set(basis[i]):
            if symbolic:
                state = sympy.Matrix.zeros(len(basis), 1)
            else:
                state = np.zeros((len(basis), 1), dtype=np.complex128)
            state[i, 0] = 1
            return state