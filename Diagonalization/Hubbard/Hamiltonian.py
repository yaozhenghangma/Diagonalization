import numpy as np
import scipy


class Hubbard:
    def __init__(self, num_sites=1, num_orbs=None, num_electrons=1):
        if num_orbs is None:
            num_orbs = [1]
        self.num_sites = num_sites
        self.num_orbs = num_orbs
        self.num_electrons = num_electrons
        self.__total_orbs = np.sum(num_orbs)
        self.Hamiltonian, self.dimension = self.__InitHamiltonian()

    def __InitHamiltonian(self):
        dim = 0
        for i in range(0, self.num_electrons+1):
            dim += (scipy.special.comb(self.__total_orbs, i, exact=True) *
                    scipy.special.comb(self.__total_orbs, self.num_electrons-i, exact=True))
        return np.zeros((dim, dim)), dim
