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

    def SOC_d5(self, lambda_value, shift=0):
        s3 = np.sqrt(3)
        # we assume the orbitals are ordered as d3z2-r2 dx2-y2 dyz dxz dxy
        soc_matrix = lambda_value/2 * np.array(
            [
                [     0,     0,      0,       0,      0,    -1j,      0,     -1,     2j,      0],
                [     0,     0,      0,       0,    -1j,      0,      1,      0,      0,    -2j],
                [     0,     0,      0,       0,      0, -1j*s3,      0,     s3,      0,      0],
                [     0,     0,      0,       0, -1j*s3,      0,    -s3,      0,      0,      0],
                [     0,    1j,      0,   1j*s3,      0,      0,    -1j,      0,      0,      1],
                [    1j,     0,  1j*s3,       0,      0,      0,      0,     1j,     -1,      0],
                [     0,     1,      0,     -s3,     1j,      0,      0,      0,      0,    -1j],
                [    -1,     0,     s3,       0,      0,    -1j,      0,      0,    -1j,      0],
                [   -2j,     0,      0,       0,      0,     -1,      0,     1j,      0,      0],
                [     0,    2j,      0,       0,      1,      0,     1j,      0,      0,      0]
            ],
            dtype=np.complex128)

        self.Hamiltonian[
            np.repeat(np.expand_dims(np.arange(shift, shift+10),0), 10, axis=0).T,
            np.repeat(np.expand_dims(np.arange(shift, shift+10),0), 10, axis=0)] += soc_matrix

    def Zeeman_d5(self, B=[0,0,0], shift=0):
        # magnetic field in the unit of hbar/muB
        gs = 2.0023193
        gl = 1.0
        Sz = np.array([
            [1.0, 0.0],
            [0.0, -1.0]
        ]) * 0.5
        Sx = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ]) * 0.5
        Sy = np.array([
            [0.0, 1.0],
            [-1.0, 0.0]
        ]) *(-0.5j)
        SB = (Sx * B[0] + Sy * B[1] + Sz * B[2]) * gs

        s3 = np.sqrt(3)
        Lz = np.array(
            [
                [     0,     0,      0,       0,      0,      0,      0,      0,    -2j,      0],
                [     0,     0,      0,       0,      0,      0,      0,      0,      0,    -2j],
                [     0,     0,      0,       0,      0,      0,      0,      0,      0,      0],
                [     0,     0,      0,       0,      0,      0,      0,      0,      0,      0],
                [     0,     0,      0,       0,      0,      0,     1j,      0,      0,      0],
                [     0,     0,      0,       0,      0,      0,      0,     1j,      0,      0],
                [     0,     0,      0,       0,    -1j,      0,      0,      0,      0,      0],
                [     0,     0,      0,       0,      0,    -1j,      0,      0,      0,      0],
                [    2j,     0,      0,       0,      0,      0,      0,      0,      0,      0],
                [     0,    2j,      0,       0,      0,      0,      0,      0,      0,      0]
            ],
            dtype=np.complex128)
        Lx = np.array(
            [
                [     0,     0,      0,       0,     1j,      0,      0,      0,      0,      0],
                [     0,     0,      0,       0,      0,     1j,      0,      0,      0,      0],
                [     0,     0,      0,       0,  s3*1j,      0,      0,      0,      0,      0],
                [     0,     0,      0,       0,      0,  s3*1j,      0,      0,      0,      0],
                [   -1j,     0, -s3*1j,       0,      0,      0,      0,      0,      0,      0],
                [     0,   -1j,      0,  -s3*1j,      0,      0,      0,      0,      0,      0],
                [     0,     0,      0,       0,      0,      0,      0,      0,     1j,      0],
                [     0,     0,      0,       0,      0,      0,      0,      0,      0,     1j],
                [     0,     0,      0,       0,      0,      0,    -1j,      0,      0,      0],
                [     0,     0,      0,       0,      0,      0,      0,    -1j,      0,      0]
            ],
            dtype=np.complex128)
        Ly = np.array(
            [
                [     0,     0,      0,       0,      0,      0,     1j,      0,      0,      0],
                [     0,     0,      0,       0,      0,      0,      0,     1j,      0,      0],
                [     0,     0,      0,       0,      0,      0, -s3*1j,      0,      0,      0],
                [     0,     0,      0,       0,      0,      0,      0, -s3*1j,      0,      0],
                [     0,     0,      0,       0,      0,      0,      0,      0,    -1j,      0],
                [     0,     0,      0,       0,      0,      0,      0,      0,      0,    -1j],
                [   -1j,     0,  s3*1j,       0,      0,      0,      0,      0,      0,      0],
                [     0,   -1j,      0,   s3*1j,      0,      0,      0,      0,      0,      0],
                [     0,     0,      0,       0,     1j,      0,      0,      0,      0,      0],
                [     0,     0,      0,       0,      0,     1j,      0,      0,      0,      0]
            ],
            dtype=np.complex128)
        LB = (Lx * B[0] + Ly * B[1] + Lz * B[2]) * gl

        for i in range(0, 5):
            self.Hamiltonian[
                np.repeat(np.expand_dims(np.arange(shift+2*i, shift+2*i+2),0), 2, axis=0).T,
                np.repeat(np.expand_dims(np.arange(shift+2*i, shift+2*i+2),0), 2, axis=0)
            ] -= SB
        self.Hamiltonian[
            np.repeat(np.expand_dims(np.arange(shift, shift+10),0), 10, axis=0).T,
            np.repeat(np.expand_dims(np.arange(shift, shift+10),0), 10, axis=0)] -= LB