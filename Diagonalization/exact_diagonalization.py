import numpy as np

def ed(hamiltonian):
    eig_values, eig_vectors = np.linalg.eig(hamiltonian)
    index = eig_values.argsort() # Ascending
    # index = eig_values.argsort()[::-1]  # Descending
    eig_values = eig_values[index]
    eig_vectors = eig_vectors[:,index]
    return eig_values, eig_vectors