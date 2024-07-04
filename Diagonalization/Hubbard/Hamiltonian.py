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

    def SOCd7(self, lam=0, shift=0):
        soc_basis = [
            {0+shift,        2+shift,         4+shift},
            {0+shift,        2+shift,         6+shift},
            {0+shift,        2+shift,         8+shift},
            {0+shift,        2+shift,         4+shift+1},
            {0+shift,        2+shift,         6+shift+1},
            {0+shift,        2+shift,         8+shift+1},
            {0+shift,        2+shift+1,       4+shift},
            {0+shift,        2+shift+1,       6+shift},
            {0+shift,        2+shift+1,       8+shift},
            {0+shift,        2+shift+1,       4+shift+1},
            {0+shift,        2+shift+1,       6+shift+1},
            {0+shift,        2+shift+1,       8+shift+1},
            {0+shift+1,      2+shift,         4+shift},
            {0+shift+1,      2+shift,         6+shift},
            {0+shift+1,      2+shift,         8+shift},
            {0+shift+1,      2+shift,         4+shift+1},
            {0+shift+1,      2+shift,         6+shift+1},
            {0+shift+1,      2+shift,         8+shift+1},
            {0+shift+1,      2+shift+1,       4+shift},
            {0+shift+1,      2+shift+1,       6+shift},
            {0+shift+1,      2+shift+1,       8+shift},
            {0+shift+1,      2+shift+1,       4+shift+1},
            {0+shift+1,      2+shift+1,       6+shift+1},
            {0+shift+1,      2+shift+1,       8+shift+1},
        ]
        site_orbitals = {
            0+shift,
            2+shift,
            4+shift,
            6+shift,
            8+shift,
            0 + shift+1,
            2 + shift+1,
            4 + shift+1,
            6 + shift+1,
            8 + shift+1
        }
        s3 = np.sqrt(3)
        s3j = np.sqrt(3) * 1j
        s32 = np.sqrt(3)*2
        #soc_matrix = lam * np.array(
        #    [
        #        #[     1,     2,     3,     4,     5,     6,     7,     8,     9,    10,    11,    12,    13,    14,    15,    16,    17,    18,    19,    20,    21,    22,    23,    24]
        #        [     0, -3j/2,     0,     0,     0,  s3/2,     0,     0,  s3/2,     0,     0,     0,     0,     0,  s3/2,     0,     0,     0,     0,     0,     0,     0,     0,     0],
        #        [  3j/2,     0,     0,     0,     0,-s3j/2,     0,     0,-s3j/2,     0,     0,     0,     0,     0,-s3j/2,     0,     0,     0,     0,     0,     0,     0,     0,     0],
        #        [     0,     0,     0, -s3/2, s3j/2,     0, -s3/2, s3j/2,     0,     0,     0,     0, -s3/2, s3j/2,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
        #        [     0,     0,-1/s32,     0, -1j/2,     0,     0,     0,     0,     0,     0,   2/3,     0,     0,     0,     0,     0,   1/6,     0,     0,   1/6,     0,     0,     0],
        #        [     0,     0,-1j/s32, 1j/2,     0,     0,     0,     0,     0,     0,     0, -2j/3,     0,     0,     0,     0,     0, -1j/6,     0,     0, -1j/6,     0,     0,     0],
        #        [ 1/s32,1j/s32,     0,     0,     0,     0,     0,     0,     0,  -2/3,  2j/3,     0,     0,     0,     0,  -1/6,  1j/6,     0,  -1/6,  1j/6,     0,     0,     0,     0],
        #        [     0,     0,-1/s32,     0,     0,     0,     0, -1j/2,     0,     0,     0,   1/6,     0,     0,     0,     0,     0,   2/3,     0,     0,   1/6,     0,     0,     0],
        #        [     0,     0,-1j/s32,    0,     0,     0,  1j/2,     0,     0,     0,     0, -1j/6,     0,     0,     0,     0,     0, -2j/3,     0,     0, -1j/6,     0,     0,     0],
        #        [ 1/s32,1j/s32,     0,     0,     0,     0,     0,     0,     0,  -1/6,  1j/6,     0,     0,     0,     0,  -2/3,  2j/3,     0,  -1/6,  1j/6,     0,     0,     0,     0],
        #        [     0,     0,     0,     0,     0,  -2/3,     0,     0,  -1/6,     0,  1j/2,     0,     0,     0,  -1/6,     0,     0,     0,     0,     0,     0,     0,     0, 1/s32],
        #        [     0,     0,     0,     0,     0, -2j/3,     0,     0, -1j/6, -1j/2,     0,     0,     0,     0, -1j/6,     0,     0,     0,     0,     0,     0,     0,     0,-1j/s32],
        #        [     0,     0,     0,   2/3,  2j/3,     0,   1/6,  1j/6,     0,     0,     0,     0,   1/6,  1j/6,     0,     0,     0,     0,     0,     0,     0,-1/s32,1j/s32,     0],
        #        [     0,     0,-1/s32,     0,     0,     0,     0,     0,     0,     0,     0,   1/6,     0, -1j/2,     0,     0,     0,   1/6,     0,     0,   2/3,     0,     0,     0],
        #        [     0,     0,-1j/s32,    0,     0,     0,     0,     0,     0,     0,     0, -1j/6,  1j/2,     0,     0,     0,     0, -1j/6,     0,     0, -2j/3,     0,     0,     0],
        #        [ 1/s32,1j/s32,     0,     0,     0,     0,     0,     0,     0,  -1/6,  1j/6,     0,     0,     0,     0,  -1/6,  1j/6,     0,  -2/3,  2j/3,     0,     0,     0,     0],
        #        [     0,     0,     0,     0,     0,  -1/6,     0,     0,  -2/3,     0,     0,     0,     0,     0,  -1/6,     0,  1j/2,     0,     0,     0,     0,     0,     0, 1/s32],
        #        [     0,     0,     0,     0,     0, -1j/6,     0,     0, -2j/3,     0,     0,     0,     0,     0, -1j/6, -1j/2,     0,     0,     0,     0,     0,     0,     0,-1j/s32],
        #        [     0,     0,     0,   1/6,  1j/6,     0,   2/3,  2j/3,     0,     0,     0,     0,   1/6,  1j/6,     0,     0,     0,     0,     0,     0,     0,-1/s32,1j/s32,     0],
        #        [     0,     0,     0,     0,     0,  -1/6,     0,     0,  -1/6,     0,     0,     0,     0,     0,  -2/3,     0,     0,     0,     0,  1j/2,     0,     0,     0, 1/s32],
        #        [     0,     0,     0,     0,     0, -1j/6,     0,     0, -1j/6,     0,     0,     0,     0,     0, -2j/3,     0,     0,     0, -1j/2,     0,     0,     0,     0,-1j/s32],
        #        [     0,     0,     0,   1/6,  1j/6,     0,   1/6,  1j/6,     0,     0,     0,     0,   2/3,  2j/3,     0,     0,     0,     0,     0,     0,     0,-1/s32,1j/s32,     0],
        #        [     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0, -s3/2,     0,     0,     0,     0,     0, -s3/2,     0,     0, -s3/2,     0,  3j/2,     0],
        #        [     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,-s3j/2,     0,     0,     0,     0,     0,-s3j/2,     0,     0,-s3j/2, -3j/2,     0,     0],
        #        [     0,     0,     0,     0,     0,     0,     0,     0,     0,  s3/2, s3j/2,     0,     0,     0,     0,  s3/2, s3j/2,     0,  s3/2, s3j/2,     0,     0,     0,     0]],
        #    dtype=np.complex128)
        soc_matrix = lam * np.array(
            [
                # [     1,     2,     3,     4,     5,     6,     7,     8,     9,    10,    11,    12,    13,    14,    15,    16,    17,    18,    19,    20,    21,    22,    23,    24]
                [      0, -3j/2,     0,     0,     0,  -1/2,     0,     0,  -1/2,     0,     0,     0,     0,     0,  -1/2,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                [   3j/2,     0,     0,     0,     0,  1j/2,     0,     0,  1j/2,     0,     0,     0,     0,     0,  1j/2,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                [      0,     0,     0,   1/2, -1j/2,     0,   1/2, -1j/2,     0,     0,     0,     0,   1/2, -1j/2,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                [      0,     0,   1/2,     0, -1j/2,     0,     0,     0,     0,     0,     0,   2/3,     0,     0,     0,     0,     0,   1/6,     0,     0,   1/6,     0,     0,     0],
                [      0,     0,  1j/2,  1j/2,     0,     0,     0,     0,     0,     0,     0, -2j/3,     0,     0,     0,     0,     0, -1j/6,     0,     0, -1j/6,     0,     0,     0],
                [   -1/2, -1j/2,     0,     0,     0,     0,     0,     0,     0,  -2/3,  2j/3,     0,     0,     0,     0,  -1/6,  1j/6,     0,  -1/6,  1j/6,     0,     0,     0,     0],
                [      0,     0,   1/2,     0,     0,     0,     0, -1j/2,     0,     0,     0,   1/6,     0,     0,     0,     0,     0,   2/3,     0,     0,   1/6,     0,     0,     0],
                [      0,     0,  1j/2,     0,     0,     0,  1j/2,     0,     0,     0,     0, -1j/6,     0,     0,     0,     0,     0, -2j/3,     0,     0, -1j/6,     0,     0,     0],
                [   -1/2, -1j/2,     0,     0,     0,     0,     0,     0,     0,  -1/6,  1j/6,     0,     0,     0,     0,  -2/3,  2j/3,     0,  -1/6,  1j/6,     0,     0,     0,     0],
                [      0,     0,     0,     0,     0,  -2/3,     0,     0,  -1/6,     0,  1j/2,     0,     0,     0,  -1/6,     0,     0,     0,     0,     0,     0,     0,     0,  -1/2],
                [      0,     0,     0,     0,     0, -2j/3,     0,     0, -1j/6, -1j/2,     0,     0,     0,     0, -1j/6,     0,     0,     0,     0,     0,     0,     0,     0,  1j/2],
                [      0,     0,     0,   2/3,  2j/3,     0,   1/6,  1j/6,     0,     0,     0,     0,   1/6,  1j/6,     0,     0,     0,     0,     0,     0,     0,   1/2, -1j/2,     0],
                [      0,     0,   1/2,     0,     0,     0,     0,     0,     0,     0,     0,   1/6,     0, -1j/2,     0,     0,     0,   1/6,     0,     0,   2/3,     0,     0,     0],
                [      0,     0,  1j/2,     0,     0,     0,     0,     0,     0,     0,     0, -1j/6,  1j/2,     0,     0,     0,     0, -1j/6,     0,     0, -2j/3,     0,     0,     0],
                [   -1/2, -1j/2,     0,     0,     0,     0,     0,     0,     0,  -1/6,  1j/6,     0,     0,     0,     0,  -1/6,  1j/6,     0,  -2/3,  2j/3,     0,     0,     0,     0],
                [      0,     0,     0,     0,     0,  -1/6,     0,     0,  -2/3,     0,     0,     0,     0,     0,  -1/6,     0,  1j/2,     0,     0,     0,     0,     0,     0,  -1/2],
                [      0,     0,     0,     0,     0, -1j/6,     0,     0, -2j/3,     0,     0,     0,     0,     0, -1j/6, -1j/2,     0,     0,     0,     0,     0,     0,     0,  1j/2],
                [      0,     0,     0,   1/6,  1j/6,     0,   2/3,  2j/3,     0,     0,     0,     0,   1/6,  1j/6,     0,     0,     0,     0,     0,     0,     0,   1/2, -1j/2,     0],
                [      0,     0,     0,     0,     0,  -1/6,     0,     0,  -1/6,     0,     0,     0,     0,     0,  -2/3,     0,     0,     0,     0,  1j/2,     0,     0,     0,  -1/2],
                [      0,     0,     0,     0,     0, -1j/6,     0,     0, -1j/6,     0,     0,     0,     0,     0, -2j/3,     0,     0,     0, -1j/2,     0,     0,     0,     0,  1j/2],
                [      0,     0,     0,   1/6,  1j/6,     0,   1/6,  1j/6,     0,     0,     0,     0,   2/3,  2j/3,     0,     0,     0,     0,     0,     0,     0,   1/2, -1j/2,     0],
                [      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,   1/2,     0,     0,     0,     0,     0,   1/2,     0,     0,   1/2,     0,  3j/2,     0],
                [      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,  1j/2,     0,     0,     0,     0,     0,  1j/2,     0,     0,  1j/2, -3j/2,     0,     0],
                [      0,     0,     0,     0,     0,     0,     0,     0,     0,  -1/2, -1j/2,     0,     0,     0,     0,  -1/2, -1j/2,     0,  -1/2, -1j/2,     0,     0,     0,     0]
            ],
            dtype=np.complex128)
        #count = 0
        for i_soc in range(0,24):
            for j_soc in range(0, 24):
                if soc_matrix[i_soc, j_soc] != 0:
                    for i in range(0, self.dimension):
                        #print(self.basis[i], soc_basis[i_soc], list(set(self.basis[i])) == soc_basis[i_soc])
                        if set(self.basis[i]).intersection(site_orbitals) == soc_basis[i_soc]:
                            for j in range(0, self.dimension):
                                if set(self.basis[j]).intersection(site_orbitals) == soc_basis[j_soc] and set(self.basis[i]).difference(site_orbitals) == set(self.basis[j]).difference(site_orbitals):
                                    #count += 1
                                    self.Hamiltonian[i, j] += soc_matrix[i_soc, j_soc]
                                    break
                            #break
        #print(count)


    def Hopping(self, hopping_matrix, sign=True):
        for i in range(0, self.dimension):
            for create in list(set(self.basis[i])):
                self.Hamiltonian[i, i] += hopping_matrix[create, create]
            for j in range(i+1, self.dimension):
                diff_set = list(set(self.basis[i]).symmetric_difference(set(self.basis[j])))
                if len(diff_set) == 2:
                    create = list(set(self.basis[i]).difference(set(self.basis[j])))[0]
                    annihi = list(set(self.basis[j]).difference(set(self.basis[i])))[0]
                    if sign:
                        if self.basis[i][0] == self.basis[j][0] or self.basis[i][1] == self.basis[j][1]:
                            self.Hamiltonian[i, j] += hopping_matrix[create, annihi]
                            self.Hamiltonian[j, i] += hopping_matrix[annihi, create]
                        else:
                            self.Hamiltonian[i, j] += hopping_matrix[create, annihi]
                            self.Hamiltonian[j, i] += hopping_matrix[annihi, create]
                    else:
                        si = False
                        for i_bi in self.basis[i]:
                            if i_bi == create:
                                break
                            else:
                                si = not si
                        for i_bj in self.basis[j]:
                            if i_bj == annihi:
                                break
                            else:
                                si = not si
                        if si:
                            self.Hamiltonian[i, j] -= hopping_matrix[create, annihi]
                            self.Hamiltonian[j, i] -= hopping_matrix[annihi, create]
                        else:
                            self.Hamiltonian[i, j] += hopping_matrix[create, annihi]
                            self.Hamiltonian[j, i] += hopping_matrix[annihi, create]

    def Hubbard(self, Hubbard_U=0, Hund_J=0, num_orbs=0, shift=0):
        if self.__symbolic:
            Hubbard_U = sympy.symbols('U')
            Hund_J = sympy.symbols("J_H")
            Hubbard_U_prime = Hubbard_U - 2*Hund_J
            Hubbard_U_prime_minus_Hund_j = Hubbard_U_prime - Hund_J
        else:
            Hubbard_U_prime = Hubbard_U - 2*Hund_J
            Hubbard_U_prime_minus_Hund_j = Hubbard_U_prime - Hund_J
            #Hubbard_U_prime_minus_Hund_j = Hubbard_U_prime
            #Hund_J = 0
        intra_orbital_list = IntraOrbital(num_orbs, shift)
        inter_orbital_list = InterOrbital(num_orbs, shift)
        inter_orbital_hund_list = InterOrbitalHund(num_orbs, shift)
        intra_orbital_annihilation, intra_orbital_creation = IntraOrbitalHoppingHund(num_orbs, shift)
        inter_orbital_annihilation, inter_orbital_creation = InterOrbitalHoppingHund(num_orbs, shift)
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
                #print(i, j)
                basis_n = set(self.basis[j])
                if len(basis.symmetric_difference(basis_n)) == 4:
                    annihi = basis_n.difference(basis)
                    create = basis.difference(basis_n)
                    # intra orbital
                    for annihilation, creation in zip(intra_orbital_annihilation, intra_orbital_creation):
                        #print(annihilation, creation)
                        if create == creation and annihi == annihilation:
                            #print(create)
                            sign = False
                            for i_create in sorted(list(create), reverse=True):
                                for i_bi in self.basis[i]:
                                    if i_bi == i_create:
                                        break
                                    else:
                                        sign = not sign
                            for i_annihi in sorted(list(annihi), reverse=True):
                                for i_bj in self.basis[j]:
                                    if i_bj == i_annihi:
                                        break
                                    else:
                                        sign = not sign
                            if sign:
                                self.Hamiltonian[i, j] += Hund_J
                            else:
                                self.Hamiltonian[i, j] -= Hund_J

                        if create == annihilation and annihi == creation:
                            #print(create)
                            sign = False
                            for i_create in sorted(list(create), reverse=True):
                                for i_bj in self.basis[j]:
                                    if i_bj == i_create:
                                        break
                                    else:
                                        sign = not sign
                            for i_annihi in sorted(list(annihi), reverse=True):
                                for i_bi in self.basis[i]:
                                    if i_bi == i_annihi:
                                        break
                                    else:
                                        sign = not sign
                            if sign:
                                self.Hamiltonian[j, i] += Hund_J
                            else:
                                self.Hamiltonian[j, i] -= Hund_J

                    # inter orbital
                    for annihilation, creation in zip(inter_orbital_annihilation, inter_orbital_creation):
                        if create == creation and annihi == annihilation:
                            sign = False
                            for i_create in sorted(list(create), reverse=True):
                                for i_bi in self.basis[i]:
                                    if i_bi == i_create:
                                        break
                                    else:
                                        sign = not sign
                            for i_annihi in sorted(list(annihi), reverse=True):
                                for i_bj in self.basis[j]:
                                    if i_bj == i_annihi:
                                        break
                                    else:
                                        sign = not sign
                            if sign:
                                self.Hamiltonian[i, j] -= Hund_J
                            else:
                                self.Hamiltonian[i, j] += Hund_J

                        if create == annihilation and annihi == creation:
                            sign = False
                            for i_create in sorted(list(create), reverse=True):
                                for i_bj in self.basis[j]:
                                    if i_bj == i_create:
                                        break
                                    else:
                                        sign = not sign
                            for i_annihi in sorted(list(annihi), reverse=True):
                                for i_bi in self.basis[i]:
                                    if i_bi == i_annihi:
                                        break
                                    else:
                                        sign = not sign
                            if sign:
                                self.Hamiltonian[j, i] -= Hund_J
                            else:
                                self.Hamiltonian[j, i] += Hund_J

    def Hubbard_ligand(self, Hubbard_U=0, Hund_J=0):
        #FIXME: rearrange orbitals
        Hubbard_U_prime = Hubbard_U - 2 * Hund_J
        Hubbard_U_prime_minus_Hund_j = Hubbard_U_prime - Hund_J

        shift = self.num_sites * self.num_orbs_per_site
        intra_orbital_list = IntraOrbital(self.num_ligands, self.num_orbs_per_ligand, self.__total_orbs, shift=shift)
        inter_orbital_list = InterOrbital(self.num_ligands, self.num_orbs_per_ligand, self.__total_orbs, shift=shift)
        inter_orbital_hund_list = InterOrbitalHund(self.num_ligands,
                                                   self.num_orbs_per_ligand,
                                                   self.__total_orbs,
                                                   shift=shift)
        intra_orbital_annihilation, intra_orbital_creation = IntraOrbitalHoppingHund(self.num_ligands,
                                                                                     self.num_orbs_per_ligand,
                                                                                     self.__total_orbs,
                                                                                     shift=shift)
        inter_orbital_annihilation, inter_orbital_creation = InterOrbitalHoppingHund(self.num_ligands,
                                                                                     self.num_orbs_per_ligand,
                                                                                     self.__total_orbs,
                                                                                     shift=shift)

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

    def HubbardCentralAndLigand(self, Hubbard_U=0):
        #FIXME: rearrange orbitals
        num_central_orbs = self.num_sites * self.num_orbs_per_site
        num_ligand_orbs = self.num_ligands * self.num_orbs_per_ligand
        central_orbs = list(range(0, num_central_orbs))
        ligand_orbs = list(range(num_central_orbs, num_central_orbs + num_ligand_orbs))
        repulsion_list = CentralAndLigand(central_orbs, ligand_orbs, self.__total_orbs)
        for i in range(0, self.dimension):
            basis = set(self.basis[i])
            # intra orbital
            for repulsion in repulsion_list:
                if repulsion.issubset(basis):
                    self.Hamiltonian[i, i] += Hubbard_U