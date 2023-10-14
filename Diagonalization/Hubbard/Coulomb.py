import numpy as np

def IntraOrbital(num_sites, num_orbs_per_site, total_orbs):
    repulsion_list = []
    for i in range(0, num_orbs_per_site):
        for j in range(0, num_sites):
            repulsion_list.append({i+j*num_orbs_per_site, i+j*num_orbs_per_site+total_orbs})
    return repulsion_list

def InterOrbital(num_sites, num_orbs_per_site, total_orbs):
    repulsion_list = []
    for i in range(0, num_orbs_per_site):
        for j in range(i+1, num_orbs_per_site):
            for k in range(0, num_sites):
                repulsion_list.append({i+k*num_orbs_per_site, j+k*num_orbs_per_site+total_orbs})
                repulsion_list.append({j+k*num_orbs_per_site, i+k*num_orbs_per_site+total_orbs})
    return repulsion_list

def InterOrbitalHund(num_sites, num_orbs_per_site, total_orbs):
    repulsion_list = []
    for i in range(0, num_orbs_per_site):
        for j in range(i+1, num_orbs_per_site):
            for k in range(0, num_sites):
                repulsion_list.append({i+k*num_orbs_per_site, j+k*num_orbs_per_site})
                repulsion_list.append({i+k*num_orbs_per_site+total_orbs, j+k*num_orbs_per_site+total_orbs})
    return repulsion_list

def IntraOrbitalHoppingHund(num_sites, num_orbs_per_site, total_orbs):
    annihilation_list = []
    creation_list = []
    for i in range(0, num_orbs_per_site):
        for j in range(i+1, num_orbs_per_site):
            for k in range(0, num_sites):
                annihilation_list.append({i+k*num_orbs_per_site, j+k*num_orbs_per_site+total_orbs})
                creation_list.append({i+k*num_orbs_per_site+total_orbs, j+k*num_orbs_per_site})

                annihilation_list.append({j+k*num_orbs_per_site, i+k*num_orbs_per_site+total_orbs})
                creation_list.append({j+k*num_orbs_per_site+total_orbs, i+k*num_orbs_per_site})

    return annihilation_list, creation_list

def InterOrbitalHoppingHund(num_sites, num_orbs_per_site, total_orbs):
    annihilation_list = []
    creation_list = []
    for i in range(0, num_orbs_per_site):
        for j in range(i+1, num_orbs_per_site):
            for k in range(0, num_sites):
                annihilation_list.append({i+k*num_orbs_per_site, i+k*num_orbs_per_site+total_orbs})
                creation_list.append({j+k*num_orbs_per_site+total_orbs, j+k*num_orbs_per_site})

                annihilation_list.append({j+k*num_orbs_per_site, j+k*num_orbs_per_site+total_orbs})
                creation_list.append({i+k*num_orbs_per_site+total_orbs, i+k*num_orbs_per_site})

    return annihilation_list, creation_list