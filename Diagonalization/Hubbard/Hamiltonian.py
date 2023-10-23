import itertools
import numpy as np
import scipy
import sympy
from Diagonalization.Hubbard.Coulomb import *


class Hubbard:
    def __init__(self, num_sites=1, num_orbs_per_site=1, num_electrons=1, symbolic=False):
        self.__ligand = False
        self.num_sites = num_sites
        self.num_orbs_per_site = num_orbs_per_site
        self.num_orbs = np.full(num_sites, num_orbs_per_site)
        self.num_electrons = num_electrons
        self.__total_orbs = num_orbs_per_site * num_sites
        self.__symbolic = symbolic
        self.Hamiltonian, self.dimension = self.__InitHamiltonian()
        self.basis = self.__InitBasis()

    def Ligand(self, num_ligands=1, num_orbs_per_ligand=1):
        if not self.__ligand:
            self.__ligand = True
            self.num_ligands = num_ligands
            self.num_orbs_per_ligand = num_orbs_per_ligand
            self.num_orbs = np.hstack((self.num_orbs, np.full(num_ligands, num_orbs_per_ligand)))
            self.__total_orbs += num_orbs_per_ligand * num_ligands
            self.Hamiltonian, self.dimension = self.__InitHamiltonian()
            self.basis = self.__InitBasis()


    def __InitHamiltonian(self):
        dim = scipy.special.comb(self.__total_orbs*2, self.num_electrons, exact=True)
        if self.__symbolic:
            return sympy.Matrix.zeros(dim, dim), dim
        else:
            return np.zeros((dim, dim), dtype=np.complex128), dim

    def __InitBasis(self):
        orbitals_list = list(range(0, self.__total_orbs*2))
        return list(itertools.combinations(orbitals_list, self.num_electrons))

    def OneSite(self, on_site_energy):
        for i in range(0, self.dimension):
            self.Hamiltonian[i, i] += np.sum(on_site_energy[list(self.basis[i])])

    def Hopping(self, hopping_matrix):
        for i in range(0, self.dimension):
            for j in range(i+1, self.dimension):
                diff_set = list(set(self.basis[i]).symmetric_difference(set(self.basis[j])))
                if len(diff_set) == 2:
                    create = list(set(self.basis[i]).difference(set(self.basis[j])))[0]
                    annihi = list(set(self.basis[j]).difference(set(self.basis[i])))[0]
                    if self.basis[i][0] == self.basis[j][0] or self.basis[i][1] == self.basis[j][1]:
                        self.Hamiltonian[i, j] += hopping_matrix[create, annihi]
                        self.Hamiltonian[j, i] += hopping_matrix[annihi, create]
                    else:
                        self.Hamiltonian[i, j] -= hopping_matrix[create, annihi]
                        self.Hamiltonian[j, i] -= hopping_matrix[annihi, create]

    def Hubbard(self, Hubbard_U=0, Hund_J=0):
        if self.__symbolic:
            Hubbard_U = sympy.symbols('U')
            Hund_J = sympy.symbols("J_H")
            Hubbard_U_prime = Hubbard_U - 2*Hund_J
            Hubbard_U_prime_minus_Hund_j = Hubbard_U_prime - Hund_J
        else:
            Hubbard_U_prime = Hubbard_U - 2*Hund_J
            Hubbard_U_prime_minus_Hund_j = Hubbard_U_prime - Hund_J
        intra_orbital_list = IntraOrbital(self.num_sites, self.num_orbs_per_site, self.__total_orbs)
        inter_orbital_list = InterOrbital(self.num_sites, self.num_orbs_per_site, self.__total_orbs)
        inter_orbital_hund_list = InterOrbitalHund(self.num_sites, self.num_orbs_per_site, self.__total_orbs)
        intra_orbital_annihilation, intra_orbital_creation = IntraOrbitalHoppingHund(self.num_sites, self.num_orbs_per_site, self.__total_orbs)
        inter_orbital_annihilation, inter_orbital_creation = InterOrbitalHoppingHund(self.num_sites, self.num_orbs_per_site, self.__total_orbs)
        for i in range(0, self.dimension):
            basis = set(self.basis[i])
            # intra orbital
            for repulsion in intra_orbital_list:
                if repulsion.issubset(basis):
                    self.Hamiltonian[i, i] += Hubbard_U

            # inter orbital
            for repulsion in inter_orbital_list:
                if repulsion.issubset(basis):
                    self.Hamiltonian[i, i] += Hubbard_U_prime

            # inter orbital
            for repulsion in inter_orbital_hund_list:
                if repulsion.issubset(basis):
                    self.Hamiltonian[i, i] += Hubbard_U_prime_minus_Hund_j

            for j in range(i+1, self.dimension):
                basis_n = set(self.basis[j])
                if len(basis.symmetric_difference(basis_n)) == 4:
                    annihi = basis_n.difference(basis)
                    create = basis.difference(basis_n)
                    # intra orbital
                    for annihilation, creation in zip(intra_orbital_annihilation, intra_orbital_creation):
                        if create == creation and annihi == annihilation:
                            self.Hamiltonian[i, j] += Hund_J
                        if create == annihilation and annihi == creation:
                            self.Hamiltonian[j, i] += Hund_J

                    # inter orbital
                    for annihilation, creation in zip(inter_orbital_annihilation, inter_orbital_creation):
                        if create == creation and annihi == annihilation:
                            self.Hamiltonian[i, j] += Hund_J
                        if create == annihilation and annihi == creation:
                            self.Hamiltonian[j, i] += Hund_J