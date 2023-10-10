import numpy as np

class SingleParticleHamiltonian:
    def __init__(self, num_sites, num_orbs_per_site, num_total_orbs):
        self.__total_orbs = num_total_orbs
        self.num_sites = num_sites
        self.num_orbs_per_site = num_orbs_per_site
        self.Hamiltonian = np.zeros((num_total_orbs*2, num_total_orbs*2), dtype=np.complex128)

    def OnSite(self, on_site_matrix):
        # spin up
        for i in range(0, self.num_sites):
            self.Hamiltonian[(i*self.num_orbs_per_site):(i*self.num_orbs_per_site+self.num_orbs_per_site),
            (i*self.num_orbs_per_site):(i*self.num_orbs_per_site+self.num_orbs_per_site)] = on_site_matrix

        # spin down
        for i in range(0, self.num_sites):
            self.Hamiltonian[(i*self.num_orbs_per_site+self.__total_orbs):(i*self.num_orbs_per_site+self.num_orbs_per_site+self.__total_orbs),
            (i*self.num_orbs_per_site+self.__total_orbs):(i*self.num_orbs_per_site+self.num_orbs_per_site+self.__total_orbs)] = on_site_matrix
        return np.diag(self.Hamiltonian)

    def Hopping(self, hopping_matrix, site_m, site_n):
        # spin up
        self.Hamiltonian[(site_m*self.num_orbs_per_site):(site_m*self.num_orbs_per_site+self.num_orbs_per_site),
        (site_n*self.num_orbs_per_site):(site_n*self.num_orbs_per_site+self.num_orbs_per_site)] += hopping_matrix

        # spin down
        self.Hamiltonian[(site_m*self.num_orbs_per_site+self.__total_orbs):(site_m*self.num_orbs_per_site+self.num_orbs_per_site+self.__total_orbs),
        (site_n*self.num_orbs_per_site+self.__total_orbs):(site_n*self.num_orbs_per_site+self.num_orbs_per_site+self.__total_orbs)] += hopping_matrix

    def SOC(self, lambda_value):
        # we assume the orbitals of t2g are ordered as dyz dxz dxy
        soc_matrix = lambda_value/2 * np.array([[  0,      0,    -1j,      0,      0,      1],
                                                [  0,      0,      0,     1j,     -1,      0],
                                                [ 1j,      0,      0,      0,      0,    -1j],
                                                [  0,    -1j,      0,      0,    -1j,      0],
                                                [  0,     -1,      0,     1j,      0,      0],
                                                [  1,      0,     1j,      0,      0,      0]], dtype=np.complex128)

        to = self.__total_orbs
        for i in range(0, self.num_sites):
            index_m = [[i*3,        i*3,        i*3,        i*3,        i*3,        i*3],
                       [i*3+to,     i*3+to,     i*3+to,     i*3+to,     i*3+to,     i*3+to],
                       [i*3+1,      i*3+1,      i*3+1,      i*3+1,      i*3+1,      i*3+1],
                       [i*3+1+to,   i*3+1+to,   i*3+1+to,   i*3+1+to,   i*3+1+to,   i*3+1+to],
                       [i*3+2,      i*3+2,      i*3+2,      i*3+2,      i*3+2,      i*3+2],
                       [i*3+2+to,   i*3+2+to,   i*3+2+to,   i*3+2+to,   i*3+2+to,   i*3+2+to]]

            index_n = [[i*3,        i*3+to,     i*3+1,      i*3+1+to,   i*3+2,      i*3+2+to],
                       [i*3,        i*3+to,     i*3+1,      i*3+1+to,   i*3+2,      i*3+2+to],
                       [i*3,        i*3+to,     i*3+1,      i*3+1+to,   i*3+2,      i*3+2+to],
                       [i*3,        i*3+to,     i*3+1,      i*3+1+to,   i*3+2,      i*3+2+to],
                       [i*3,        i*3+to,     i*3+1,      i*3+1+to,   i*3+2,      i*3+2+to],
                       [i*3,        i*3+to,     i*3+1,      i*3+1+to,   i*3+2,      i*3+2+to]]
            self.Hamiltonian[index_m, index_n] += soc_matrix