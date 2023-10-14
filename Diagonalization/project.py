import numpy as np


def LowEnergy(eig_values, eig_vectors, num_states):
    eig_values = eig_values[0:num_states]
    eig_vectors = eig_vectors[:, 0:num_states]
    return eig_values, eig_vectors


def ChooseLowEnergy(eig_values, eig_vectors, index_states):
    eig_values = eig_values[index_states]
    eig_vectors = eig_vectors[:, index_states]
    return eig_values, eig_vectors


def LowdinOrthonormalization(vectors):
    u, s, vh = np.linalg.svd(vectors, full_matrices=False)
    return np.matmul(u, vh)


def Project(eig_values, eig_vectors, projected_states):
    transform_U = np.zeros((eig_values.shape[0], projected_states.shape[1]), dtype=np.complex128)
    for i in range(0, eig_values.shape[0]):
        for j in range(0, projected_states.shape[1]):
            transform_U[i, j] = np.dot(eig_vectors[:, i].conj(), projected_states[:, j])
    #normalized_U, r = np.linalg.qr(transform_U)
    normalized_U = LowdinOrthonormalization(transform_U)
    normalized_U[:, 1] = -normalized_U[:, 1]    # FIXME: unknown reason, antisymmetric form is favored
    #normalized_U[:, 0] = -normalized_U[:, 0]

    projected_Hamiltonian = np.matmul(normalized_U.conj().T, np.matmul(np.diag(eig_values), normalized_U))
    return projected_Hamiltonian, normalized_U, transform_U
