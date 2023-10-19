import Diagonalization
import pytest
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=110)


def test_soc_in_t2g():
    single_particle = Diagonalization.SingleParticleHamiltonian(1, 3, 3)
    single_particle.SOC(1.0)
    print()
    print(single_particle.Hamiltonian)
    print(np.abs(single_particle.Hamiltonian-single_particle.Hamiltonian.conj().T))
    eig_values, eig_vectors = np.linalg.eig(single_particle.Hamiltonian)
    index = eig_values.real.argsort()
    print(eig_values[index])
    print(eig_vectors[:,index])