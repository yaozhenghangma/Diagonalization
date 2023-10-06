import itertools
import numpy as np
import scipy


class Hubbard:
    def __init__(self, num_sites=1, num_orbs_per_site=1, num_electrons=1):
        self.num_sites = num_sites
        self.num_orbs = np.full(num_sites, num_orbs_per_site)
        self.num_electrons = num_electrons
        self.__total_orbs = num_orbs_per_site * num_sites
        self.Hamiltonian, self.dimension = self.__InitHamiltonian()
        self.basis = self.__InitBasis()

    def __InitHamiltonian(self):
        dim = scipy.special.comb(self.__total_orbs*2, self.num_electrons, exact=True)
        return np.zeros((dim, dim)), dim

    def __InitBasis(self):
        orbitals_list = list(range(0, self.num_orbs*2))
        return list(itertools.combinations(orbitals_list, self.num_electrons))

    def OneSite(self, on_site_energy):
        for i in range(0, self.__total_orbs*2):
            self.Hamiltonian[i, i] = np.sum(on_site_energy[list(self.basis[i])])

    def Hopping(self, hopping_matrix):
        for i in range(0, self.__total_orbs*2):
            for j in range(i+1, self.__total_orbs*2):
                diff_set = list(set(self.basis[i]).symmetric_difference(set(self.basis[j])))
                if len(diff_set) == 2:
                    self.Hamiltonian[i, j] = hopping_matrix[diff_set[0], diff_set[1]]
                    self.Hamiltonian[j, i] = hopping_matrix[diff_set[0], diff_set[1]]

    def Hubbard(self):
        for i in range(0, self.__total_orbs):
            wavefunction = np.zeros(self.num_orbs*2)
            wavefunction[self.basis[i]] = 1
            spin_up = wavefunction[0:self.num_orbs]
            spin_dn = wavefunction[self.num_orbs:-1]
            pass