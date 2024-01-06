import numpy as np

def IntraOrbital(num_sites, num_orbs_per_site, total_orbs, shift=0):
    repulsion_list = []
    for i in range(0, num_orbs_per_site):
        for j in range(0, num_sites):
            repulsion_list.append({i+j*num_orbs_per_site+shift, i+j*num_orbs_per_site+total_orbs+shift})
    return repulsion_list

def InterOrbital(num_sites, num_orbs_per_site, total_orbs, shift=0):
    repulsion_list = []
    for i in range(0, num_orbs_per_site):
        for j in range(i+1, num_orbs_per_site):
            for k in range(0, num_sites):
                repulsion_list.append({i+k*num_orbs_per_site+shift, j+k*num_orbs_per_site+total_orbs+shift})
                repulsion_list.append({j+k*num_orbs_per_site+shift, i+k*num_orbs_per_site+total_orbs+shift})
    return repulsion_list

def CentralAndLigand(central_orbs, ligand_orbs, total_orbs):
    repulsion_list = []
    for ic in central_orbs:
        for il in ligand_orbs:
            repulsion_list.append({ic, il})
            repulsion_list.append({ic+total_orbs, il})
            repulsion_list.append({ic, il+total_orbs})
            repulsion_list.append({ic+total_orbs, il+total_orbs})
    return repulsion_list

def InterOrbitalHund(num_sites, num_orbs_per_site, total_orbs, shift=0):
    repulsion_list = []
    for i in range(0, num_orbs_per_site):
        for j in range(i+1, num_orbs_per_site):
            for k in range(0, num_sites):
                repulsion_list.append({i+k*num_orbs_per_site+shift, j+k*num_orbs_per_site+shift})
                repulsion_list.append({i+k*num_orbs_per_site+total_orbs+shift, j+k*num_orbs_per_site+total_orbs+shift})
    return repulsion_list

def IntraOrbitalHoppingHund(num_sites, num_orbs_per_site, total_orbs, shift=0):
    annihilation_list = []
    creation_list = []
    for i in range(0, num_orbs_per_site):
        for j in range(i+1, num_orbs_per_site):
            for k in range(0, num_sites):
                annihilation_list.append({i+k*num_orbs_per_site+shift, j+k*num_orbs_per_site+total_orbs+shift})
                creation_list.append({i+k*num_orbs_per_site+total_orbs+shift, j+k*num_orbs_per_site+shift})

                annihilation_list.append({j+k*num_orbs_per_site+shift, i+k*num_orbs_per_site+total_orbs+shift})
                creation_list.append({j+k*num_orbs_per_site+total_orbs+shift, i+k*num_orbs_per_site+shift})

    return annihilation_list, creation_list

def InterOrbitalHoppingHund(num_sites, num_orbs_per_site, total_orbs, shift=0):
    annihilation_list = []
    creation_list = []
    for i in range(0, num_orbs_per_site):
        for j in range(i+1, num_orbs_per_site):
            for k in range(0, num_sites):
                annihilation_list.append({i+k*num_orbs_per_site+shift, i+k*num_orbs_per_site+total_orbs+shift})
                creation_list.append({j+k*num_orbs_per_site+total_orbs+shift, j+k*num_orbs_per_site+shift})

                annihilation_list.append({j+k*num_orbs_per_site+shift, j+k*num_orbs_per_site+total_orbs+shift})
                creation_list.append({i+k*num_orbs_per_site+total_orbs+shift, i+k*num_orbs_per_site+shift})

    return annihilation_list, creation_list