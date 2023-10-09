import numpy as np

def LowEnergy(eig_values, eig_vectors):
    num_states = eig_values.shape[0]
    half_states = int(np.ceil(num_states/2))
    eig_values = eig_values[0:half_states]
    eig_vectors = eig_vectors[:,0:half_states]
    return eig_values, eig_vectors

def LowdinOrthonormalization(vectors):
    overlap = np.matmul(vectors.T, vectors)
    eig_values, eig_vectors = np.linalg.eigh(overlap)
    eig_values = np.diag(eig_values)
    sqrt_inv_eig = np.sqrt(np.linalg.inv(eig_values))
    transform_s = np.dot(eig_vectors, np.dot(sqrt_inv_eig, eig_vectors.T))
    return np.matmul(vectors, transform_s)


def Project(eig_values, eig_vectors, projected_states):
    transform_U = np.zeros((eig_values.shape[0], projected_states.shape[1]))
    for i in range(0, eig_values.shape[0]):
        for j in range(0, projected_states.shape[1]):
            transform_U[i, j] = np.dot(eig_vectors[:,i], projected_states[:,j])
    #normalized_U, r = np.linalg.qr(transform_U)
    normalized_U = LowdinOrthonormalization(transform_U)

    projected_Hamiltonian = np.matmul(normalized_U.conj().transpose, np.matmul(np.diag(eig_values), normalized_U))
    return projected_Hamiltonian, normalized_U
