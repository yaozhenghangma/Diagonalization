import numpy as np


def ChooseState(basis, orbs):
    for i in range(0, len(basis)):
        if orbs == set(basis[i]):
            state = np.zeros((len(basis), 1), dtype=np.complex128)
            state[i, 0] = 1
            return state