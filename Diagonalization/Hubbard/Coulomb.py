import numpy as np

def IntraOrbital(num_orbs, shift=0):
    repulsion_list = []
    for i in range(0, num_orbs*2, 2):
        repulsion_list.append({i+shift, i+shift+1})
    return repulsion_list

def InterOrbital(num_orbs, shift=0):
    repulsion_list = []
    for i in range(0, num_orbs*2, 2):
        for j in range(i+2, num_orbs*2, 2):
            repulsion_list.append({i+shift, j+1+shift})
            repulsion_list.append({j+shift, i+1+shift})
    return repulsion_list

def CentralAndLigand(central_orbs, ligand_orbs, total_orbs):
    #FIXME: rearrange orbitals
    repulsion_list = []
    for ic in central_orbs:
        for il in ligand_orbs:
            repulsion_list.append({ic, il})
            repulsion_list.append({ic+total_orbs, il})
            repulsion_list.append({ic, il+total_orbs})
            repulsion_list.append({ic+total_orbs, il+total_orbs})
    return repulsion_list

def InterOrbitalHund(num_orbs, shift=0):
    repulsion_list = []
    for i in range(0, num_orbs*2, 2):
        for j in range(i+2, num_orbs*2, 2):
            repulsion_list.append({i+shift, j+shift})
            repulsion_list.append({i+1+shift, j+1+shift})
    return repulsion_list

def IntraOrbitalHoppingHund(num_orbs, shift=0):
    annihilation_list = []
    creation_list = []
    for i in range(0, num_orbs*2, 2):
        for j in range(i+2, num_orbs*2, 2):
            annihilation_list.append({i+shift, j+1+shift})
            creation_list.append({i+1+shift, j+shift})

            annihilation_list.append({j+shift, i+1+shift})
            creation_list.append({j+1+shift, i+shift})

    return annihilation_list, creation_list

def InterOrbitalHoppingHund(num_orbs, shift=0):
    annihilation_list = []
    creation_list = []
    for i in range(0, num_orbs*2, 2):
        for j in range(i+2, num_orbs*2,2):
            annihilation_list.append({i+shift, i+1+shift})
            creation_list.append({j+1+shift, j+shift})

            annihilation_list.append({j+shift, j+1+shift})
            creation_list.append({i+1+shift, i+shift})

    return annihilation_list, creation_list