import numpy as np

def LowEnergy(eig_values, eig_vectors):
    num_states = eig_values.shape[0]
    half_states = np.ceil(num_states/2)
    eig_values = eig_values[0:half_states]
    eig_vectors = eig_vectors[:,0:half_states]
    return eig_values, eig_vectors