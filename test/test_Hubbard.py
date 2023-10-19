import Diagonalization
import pytest
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=110)


def test_Hamiltonian_build():
    hubbard = Diagonalization.Hubbard(2, 3, 2)
    assert hubbard.dimension == 66


def test_basis():
    hubbard = Diagonalization.Hubbard(2, 1, 2)
    print(hubbard.basis)


def test_single_particle():
    single_particle = Diagonalization.SingleParticleHamiltonian(2, 1, 2)
    single_particle.OnSite(np.zeros((1,1)))
    single_particle.Hopping(np.array([[-1]]), 0, 1)
    print()
    print(single_particle.Hamiltonian)


def test_two_bands():
    single_particle = Diagonalization.SingleParticleHamiltonian(2, 1, 2)
    single_particle.OnSite(np.zeros((1, 1)))
    single_particle.Hopping(np.array([[-1]]), 0, 1)
    hubbard = Diagonalization.Hubbard(2, 1, 2)
    hubbard.Hopping(single_particle.Hamiltonian)
    hubbard.Hubbard(3.0, 0.0)
    print()
    print(hubbard.Hamiltonian)


def test_one_site_two_bands():
    single_particle = Diagonalization.SingleParticleHamiltonian(1, 2, 2)
    single_particle.OnSite(np.array([[0, -1],[-1,0]]))
    hubbard = Diagonalization.Hubbard(1, 2, 2)
    hubbard.Hopping(single_particle.Hamiltonian)
    hubbard.Hubbard(3.0, 0.3)
    print()
    print(hubbard.basis)
    print(hubbard.Hamiltonian)


def test_three_bands():
    single_particle = Diagonalization.SingleParticleHamiltonian(3, 1, 3)
    single_particle.OnSite(np.zeros((1, 1)))
    single_particle.Hopping(np.array([[-0.5]]), 0, 1)
    single_particle.Hopping(np.array([[-0.5]]), 1, 2)
    single_particle.Hopping(np.array([[-1]]), 0, 2)
    hubbard = Diagonalization.Hubbard(3, 1, 2)
    hubbard.basis=[(0,1),(0,2),(1,2),(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)]
    hubbard.Hopping(single_particle.Hamiltonian)
    hubbard.Hubbard(3.0, 0.0)
    print()
    print(hubbard.basis)
    print(hubbard.Hamiltonian)


def test_one_site_three_bands():
    single_particle = Diagonalization.SingleParticleHamiltonian(1, 3, 3)
    single_particle.OnSite(np.array([[0,-0.5,-1],[-0.5,0,-0.5],[-1,-0.5,0]]))
    hubbard = Diagonalization.Hubbard(1, 3, 2)
    hubbard.basis = [(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5),
                     (3, 4), (3, 5), (4, 5)]
    hubbard.Hopping(single_particle.Hamiltonian)
    hubbard.Hubbard(3.0, 0.0)
    print()
    print(hubbard.basis)
    print(hubbard.Hamiltonian)