import numpy as np
import sympy

class SingleParticleHamiltonian:
    def __init__(self, num_sites, num_orbs_per_site, symbolic=False):
        self.__ligand = False
        self.__total_orbs = num_sites * num_orbs_per_site
        self.num_sites = num_sites
        self.num_orbs_per_site = num_orbs_per_site
        self.__symbolic = symbolic
        if symbolic:
            self.Hamiltonian = sympy.Matrix.zeros(self.__total_orbs*2, self.__total_orbs*2)
        else:
            self.Hamiltonian = np.zeros((self.__total_orbs*2, self.__total_orbs*2), dtype=np.complex128)

    def Ligand(self, num_ligands, num_orbs_per_ligand):
        if not self.__ligand:
            self.__ligand = True
            self.num_ligands = num_ligands
            self.num_orbs_per_ligand = num_orbs_per_ligand
            self.__central_orbs = self.__total_orbs
            self.__total_orbs += num_orbs_per_ligand * num_ligands
            self.Hamiltonian = np.zeros((self.__total_orbs*2, self.__total_orbs*2), dtype=np.complex128)

    def OnSite(self, on_site_matrix, shift=0):
        dim_m = on_site_matrix.shape[0] # dimension of onsite matrix of each spin state
        num_orbs = on_site_matrix.shape[0]*2
        # spin up
        self.Hamiltonian[
            np.repeat(np.expand_dims(np.arange(shift, shift+num_orbs, 2),0), dim_m, axis=0).T,
            np.repeat(np.expand_dims(np.arange(shift, shift+num_orbs, 2),0), dim_m, axis=0)] += on_site_matrix
        # spin down
        self.Hamiltonian[
            np.repeat(np.expand_dims(np.arange(shift+1, shift+num_orbs, 2),0), dim_m, axis=0).T,
            np.repeat(np.expand_dims(np.arange(shift+1, shift+num_orbs, 2),0), dim_m, axis=0)] += on_site_matrix
        return np.diag(self.Hamiltonian)

    def OnSiteLigand(self, on_site_matrix):
        # FIXME: rearrange creation operator
        # spin up
        for i in range(0, self.num_ligands):
            self.Hamiltonian[(i * self.num_orbs_per_site + self.__central_orbs):(
                    i * self.num_orbs_per_site + self.num_orbs_per_site + self.__central_orbs),
            (i * self.num_orbs_per_site + self.__central_orbs):(
                    i * self.num_orbs_per_site + self.num_orbs_per_site + self.__central_orbs)] = on_site_matrix

        # spin down
        for i in range(0, self.num_ligands):
            self.Hamiltonian[(i * self.num_orbs_per_site + self.__total_orbs + self.__central_orbs):(
                        i * self.num_orbs_per_site + self.num_orbs_per_site + self.__total_orbs + self.__central_orbs),
            (i * self.num_orbs_per_site + self.__total_orbs + self.__central_orbs):(
                        i * self.num_orbs_per_site + self.num_orbs_per_site + self.__total_orbs + self.__central_orbs)]\
                = on_site_matrix
        return np.diag(self.Hamiltonian)

    def Hopping(self, hopping_matrix, shift_m=0, shift_n=0):
        # m*n hopping matrix
        dim_m = hopping_matrix.shape[0]
        dim_n = hopping_matrix.shape[1]
        num_orbs_m = hopping_matrix.shape[0]*2
        num_orbs_n = hopping_matrix.shape[1]*2
        # spin up
        self.Hamiltonian[
            np.repeat(np.expand_dims(np.arange(shift_m, shift_m+num_orbs_m, 2),0), dim_m, axis=0).T,
            np.repeat(np.expand_dims(np.arange(shift_n, shift_n+num_orbs_n, 2),0), dim_n, axis=0)] += hopping_matrix
        self.Hamiltonian[
            np.repeat(np.expand_dims(np.arange(shift_n, shift_n+num_orbs_n, 2),0), dim_n, axis=0).T,
            np.repeat(np.expand_dims(np.arange(shift_m, shift_m+num_orbs_m, 2),0), dim_m, axis=0)] += hopping_matrix.conj().T

        # spin down
        self.Hamiltonian[
            np.repeat(np.expand_dims(np.arange(shift_m+1, shift_m+num_orbs_m, 2),0), dim_m, axis=0).T,
            np.repeat(np.expand_dims(np.arange(shift_n+1, shift_n+num_orbs_n, 2),0), dim_n, axis=0)] += hopping_matrix
        self.Hamiltonian[
            np.repeat(np.expand_dims(np.arange(shift_n+1, shift_n+num_orbs_n, 2),0), dim_n, axis=0).T,
            np.repeat(np.expand_dims(np.arange(shift_m+1, shift_m+num_orbs_m, 2),0), dim_m, axis=0)] += hopping_matrix.conj().T

    def SOC(self, lambda_value, shift=0):
        # we assume the orbitals of t2g are ordered as dyz dxz dxy
        soc_matrix = lambda_value/2 * np.array([[  0,      0,    -1j,      0,      0,      1],
                                                [  0,      0,      0,     1j,     -1,      0],
                                                [ 1j,      0,      0,      0,      0,    -1j],
                                                [  0,    -1j,      0,      0,    -1j,      0],
                                                [  0,     -1,      0,     1j,      0,      0],
                                                [  1,      0,     1j,      0,      0,      0]], dtype=np.complex128)
        self.Hamiltonian[
            np.repeat(np.expand_dims(np.arange(shift, shift+6),0), 6, axis=0).T,
            np.repeat(np.expand_dims(np.arange(shift, shift+6),0), 6, axis=0)] += soc_matrix

    def SOC_d5(self, lambda_value):
        # we assume the orbitals are ordered as d3z2-r2 dx2-y2 dyz dxz dxy
        s3 = np.sqrt(3)
        soc_matrix = lambda_value/2 * np.array(
            [
                #[     0,     0,      0,       0,      0, -s3*1j,      0,     s3,      0,      0],
                #[     0,     0,      0,       0, -s3*1j,      0,    -s3,      0,      0,      0],
                #[     0,     0,      0,       0,      0,    -1j,      0,     -1,     2j,      0],
                #[     0,     0,      0,       0,    -1j,      0,      1,      0,      0,    -2j],
                #[     0, s3*1j,      0,      1j,      0,      0,    -1j,      0,      0,      1],
                #[ s3*1j,     0,     1j,       0,      0,      0,      0,     1j,     -1,      0],
                #[     0,   -s3,      0,       1,     1j,      0,      0,      0,      0,    -1j],
                #[    s3,     0,     -1,       0,      0,    -1j,      0,      0,    -1j,      0],
                #[     0,     0,    -2j,       0,      0,     -1,      0,     1j,      0,      0],
                #[     0,     0,      0,      2j,      1,      0,     1j,      0,      0,      0]],

                [0, 0, 0, 0, 0, 0, 0, s3*1j, 0, -s3],
                [0, 0, 0, 0, 0, 0, s3*1j, 0, s3, 0],
                [0, 0, 0, 0, 2j, 0, 0, 1, 0, -1j],
                [0, 0, 0, 0, 0, -2j, -1, 0, -1j, 0],
                [0, 0, -2j, 0, 0, 0, 0, 1j, 0, 1],
                [0, 0, 0, 2j, 0, 0, 1j, 0, -1, 0],
                [0, -s3*1j, 0, -1, 0, -1j, 0, 0, 1j, 0],
                [-s3*1j, 0, 1, 0, -1j, 0, 0, 0, 0, -1j],
                [0, s3, 0, 1j, 0, -1, -1j, 0, 0, 0],
                [-s3, 0, 1j, 0, 1, 0, 0, 1j, 0, 0]
            ],
            dtype=np.complex128)

        to = self.__total_orbs
        for i in range(0, self.num_sites):
            index_m = [
                [i*5,       i*5,        i*5,        i*5,        i*5,        i*5,        i*5,        i*5,        i*5,        i*5],
                [i*5+to,    i*5+to,     i*5+to,     i*5+to,     i*5+to,     i*5+to,     i*5+to,     i*5+to,     i*5+to,     i*5+to],
                [i*5+1,     i*5+1,      i*5+1,      i*5+1,      i*5+1,      i*5+1,      i*5+1,      i*5+1,      i*5+1,      i*5+1],
                [i*5+1+to,  i*5+1+to,   i*5+1+to,   i*5+1+to,   i*5+1+to,   i*5+1+to,   i*5+1+to,   i*5+1+to,   i*5+1+to,   i*5+1+to],
                [i*5+2,     i*5+2,      i*5+2,      i*5+2,      i*5+2,      i*5+2,      i*5+2,      i*5+2,      i*5+2,      i*5+2],
                [i*5+2+to,  i*5+2+to,   i*5+2+to,   i*5+2+to,   i*5+2+to,   i*5+2+to,   i*5+2+to,   i*5+2+to,   i*5+2+to,   i*5+2+to],
                [i*5+3,     i*5+3,      i*5+3,      i*5+3,      i*5+3,      i*5+3,      i*5+3,      i*5+3,      i*5+3,      i*5+3],
                [i*5+3+to,  i*5+3+to,   i*5+3+to,   i*5+3+to,   i*5+3+to,   i*5+3+to,   i*5+3+to,   i*5+3+to,   i*5+3+to,   i*5+3+to],
                [i*5+4,     i*5+4,      i*5+4,      i*5+4,      i*5+4,      i*5+4,      i*5+4,      i*5+4,      i*5+4,      i*5+4],
                [i*5+4+to,  i*5+4+to,   i*5+4+to,   i*5+4+to,   i*5+4+to,   i*5+4+to,   i*5+4+to,   i*5+4+to,   i*5+4+to,   i*5+4+to]
            ]

            index_n = [
                [i*5,   i*5+to, i*5+1,  i*5+1+to,   i*5+2,  i*5+2+to,   i*5+3,  i*5+3+to,   i*5+4,  i*5+4+to],
                [i*5,   i*5+to, i*5+1,  i*5+1+to,   i*5+2,  i*5+2+to,   i*5+3,  i*5+3+to,   i*5+4,  i*5+4+to],
                [i*5,   i*5+to, i*5+1,  i*5+1+to,   i*5+2,  i*5+2+to,   i*5+3,  i*5+3+to,   i*5+4,  i*5+4+to],
                [i*5,   i*5+to, i*5+1,  i*5+1+to,   i*5+2,  i*5+2+to,   i*5+3,  i*5+3+to,   i*5+4,  i*5+4+to],
                [i*5,   i*5+to, i*5+1,  i*5+1+to,   i*5+2,  i*5+2+to,   i*5+3,  i*5+3+to,   i*5+4,  i*5+4+to],
                [i*5,   i*5+to, i*5+1,  i*5+1+to,   i*5+2,  i*5+2+to,   i*5+3,  i*5+3+to,   i*5+4,  i*5+4+to],
                [i*5,   i*5+to, i*5+1,  i*5+1+to,   i*5+2,  i*5+2+to,   i*5+3,  i*5+3+to,   i*5+4,  i*5+4+to],
                [i*5,   i*5+to, i*5+1,  i*5+1+to,   i*5+2,  i*5+2+to,   i*5+3,  i*5+3+to,   i*5+4,  i*5+4+to],
                [i*5,   i*5+to, i*5+1,  i*5+1+to,   i*5+2,  i*5+2+to,   i*5+3,  i*5+3+to,   i*5+4,  i*5+4+to],
                [i*5,   i*5+to, i*5+1,  i*5+1+to,   i*5+2,  i*5+2+to,   i*5+3,  i*5+3+to,   i*5+4,  i*5+4+to]]

            self.Hamiltonian[index_m, index_n] += soc_matrix

    def SOC_l2(self, lambda_value):
        soc_matrix = lambda_value / 2 * np.array([
                [0,         0, -1j, 0, 0, 0, 0, 0, 1j/2, 1 / 2],
                [0,         0, 0, 0, 0, 0, 0, 0, (1j*np.sqrt(3))/2, -(np.sqrt(3)/2)],
                [1j,        0, 0, 0, 0, 0, 0, 0, 1 / 2, -(1j/2)],
                [0,         0, 0, 0, 1j/2, -(1j/2), -((1j*np.sqrt(3)) / 2), -(1/2), 0, 0],
                [0,         0, 0, -(1j/2), 0, -(1 / 2), np.sqrt(3) / 2, 1j/2, 0, 0],
                [0,         0, 0, 1j/ 2, -(1 / 2), 0, 0, 1j, 0, 0],
                [0,         0, 0, (1j*np.sqrt(3)) / 2, np.sqrt(3) / 2, 0, 0, 0, 0, 0],
                [0,         0, 0, -(1 / 2), -(1j/2), -1j, 0, 0, 0, 0],
                [-(1j/2), -((1j*np.sqrt(3))/2), 1 / 2, 0, 0, 0, 0, 0, 0, -(1j/2)],
                [1 / 2,     -(np.sqrt(3)/2), 1j/2, 0, 0, 0, 0, 0, 1j/2, 0]
        ])
        to = self.__total_orbs
        for i in range(0, self.num_sites):
            index_n = [[i * 5, i * 5 + to, i * 5 + 1, i * 5 + 1 + to, i * 5 + 2, i * 5 + 2 + to, i * 5 + 3,
                        i * 5 + 3 + to, i * 5 + 4, i * 5 + 4 + to],
                       [i * 5, i * 5 + to, i * 5 + 1, i * 5 + 1 + to, i * 5 + 2, i * 5 + 2 + to, i * 5 + 3,
                        i * 5 + 3 + to, i * 5 + 4, i * 5 + 4 + to],
                       [i * 5, i * 5 + to, i * 5 + 1, i * 5 + 1 + to, i * 5 + 2, i * 5 + 2 + to, i * 5 + 3,
                        i * 5 + 3 + to, i * 5 + 4, i * 5 + 4 + to],
                       [i * 5, i * 5 + to, i * 5 + 1, i * 5 + 1 + to, i * 5 + 2, i * 5 + 2 + to, i * 5 + 3,
                        i * 5 + 3 + to, i * 5 + 4, i * 5 + 4 + to],
                       [i * 5, i * 5 + to, i * 5 + 1, i * 5 + 1 + to, i * 5 + 2, i * 5 + 2 + to, i * 5 + 3,
                        i * 5 + 3 + to, i * 5 + 4, i * 5 + 4 + to],
                       [i * 5, i * 5 + to, i * 5 + 1, i * 5 + 1 + to, i * 5 + 2, i * 5 + 2 + to, i * 5 + 3,
                        i * 5 + 3 + to, i * 5 + 4, i * 5 + 4 + to],
                       [i * 5, i * 5 + to, i * 5 + 1, i * 5 + 1 + to, i * 5 + 2, i * 5 + 2 + to, i * 5 + 3,
                        i * 5 + 3 + to, i * 5 + 4, i * 5 + 4 + to],
                       [i * 5, i * 5 + to, i * 5 + 1, i * 5 + 1 + to, i * 5 + 2, i * 5 + 2 + to, i * 5 + 3,
                        i * 5 + 3 + to, i * 5 + 4, i * 5 + 4 + to],
                       [i * 5, i * 5 + to, i * 5 + 1, i * 5 + 1 + to, i * 5 + 2, i * 5 + 2 + to, i * 5 + 3,
                        i * 5 + 3 + to, i * 5 + 4, i * 5 + 4 + to],
                       [i * 5, i * 5 + to, i * 5 + 1, i * 5 + 1 + to, i * 5 + 2, i * 5 + 2 + to, i * 5 + 3,
                        i * 5 + 3 + to, i * 5 + 4, i * 5 + 4 + to]
                       ]
            index_m = np.array(index_n).T.tolist()
            self.Hamiltonian[index_m, index_n] += soc_matrix