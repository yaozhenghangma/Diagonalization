import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True, linewidth=110)


def LowEnergy(eig_values, eig_vectors):
    low_states = 4
    eig_values = eig_values[0:low_states]
    eig_vectors = eig_vectors[:,0:low_states]
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
    normalized_U = LowdinOrthonormalization(transform_U)

    projected_Hamiltonian = np.matmul(normalized_U.conj().T, np.matmul(np.diag(eig_values), normalized_U))
    return projected_Hamiltonian, normalized_U


def ThreeBand(tdd, tpd, Udd, Upp, delta):
    Hamiltonian = np.zeros((15, 15))

    # hopping term
    Hamiltonian[0, 2] = tdd
    Hamiltonian[2, 0] = tdd

    Hamiltonian[3, 9] = tdd
    Hamiltonian[9, 3] = tdd

    Hamiltonian[4, 10] = tdd
    Hamiltonian[10, 4] = tdd

    Hamiltonian[5, 11] = tdd
    Hamiltonian[11, 5] = tdd

    Hamiltonian[3, 5] = tdd
    Hamiltonian[5, 3] = tdd

    Hamiltonian[6, 8] = tdd
    Hamiltonian[8, 6] = tdd

    Hamiltonian[9, 11] = tdd
    Hamiltonian[11, 9] = tdd

    Hamiltonian[12, 14] = tdd
    Hamiltonian[14, 12] = tdd

    Hamiltonian[3, 6] = tpd
    Hamiltonian[6, 3] = tpd

    Hamiltonian[4, 7] = tpd
    Hamiltonian[7, 4] = tpd

    Hamiltonian[5, 8] = tpd
    Hamiltonian[8, 5] = tpd

    Hamiltonian[9, 6] = tpd
    Hamiltonian[6, 9] = tpd

    Hamiltonian[7, 10] = tpd
    Hamiltonian[10, 7] = tpd

    Hamiltonian[8, 11] = tpd
    Hamiltonian[11, 8] = tpd

    Hamiltonian[3, 4] = tpd
    Hamiltonian[4, 3] = tpd

    Hamiltonian[7, 6] = tpd
    Hamiltonian[6, 7] = tpd

    Hamiltonian[9, 10] = tpd
    Hamiltonian[10, 9] = tpd

    Hamiltonian[4, 5] = tpd
    Hamiltonian[5, 4] = tpd

    Hamiltonian[7, 8] = tpd
    Hamiltonian[8, 7] = tpd

    Hamiltonian[10, 11] = tpd
    Hamiltonian[11, 10] = tpd

    Hamiltonian[0, 1] = tpd
    Hamiltonian[1, 0] = tpd

    Hamiltonian[1, 2] = tpd
    Hamiltonian[2, 1] = tpd

    Hamiltonian[12, 13] = tpd
    Hamiltonian[13, 12] = tpd

    Hamiltonian[13, 14] = tpd
    Hamiltonian[14, 13] = tpd

    # Hubbard term
    Hamiltonian[3, 3] = Udd
    Hamiltonian[7, 7] = Upp
    Hamiltonian[11, 11] = Udd

    # chemistry potential
    Hamiltonian[0, 0] = delta
    Hamiltonian[2, 2] = delta
    Hamiltonian[12, 12] = delta
    Hamiltonian[14, 14] = delta
    Hamiltonian[4, 4] = delta
    Hamiltonian[10, 10] = delta
    Hamiltonian[6, 6] = delta
    Hamiltonian[8, 8] = delta
    Hamiltonian[7, 7] += delta*2

    eig_values, eig_vectors = np.linalg.eig(Hamiltonian)
    index = eig_values.argsort() # Ascending
    eig_values = eig_values[index]
    eig_vectors = eig_vectors[:,index]

    eig_values, eig_vectors = LowEnergy(eig_values, eig_vectors)

    projected_states = np.zeros((15, 4))
    projected_states[5, 0] = 1
    projected_states[9, 1] = 1
    projected_states[1, 2] = 1
    projected_states[13, 3] = 1
    projected_Hamiltonian, normalized_U = Project(eig_values, eig_vectors, projected_states)
    return 2*projected_Hamiltonian[0,1]

tdd = -0.0
tpd = -0.0
Udd = 1.0
Upp = 0.8
delta = 0.5
U=1.0

perturbation1 = []
perturbation2 = []
exact_diagonalization = []
for tpd in np.arange(0, -0.2, -0.005):
    perturbation1.append(4*tpd*tpd*tpd*tpd/delta/delta * 1/Udd)
    perturbation2.append(4*tpd*tpd*tpd*tpd/delta/delta * (1/Udd+2/(2*delta+Upp)))
    exact_diagonalization.append(-ThreeBand(tdd, tpd, Udd, Upp, delta))

font = {'family' : 'Times New Roman',
'weight' : 'regular',
'size' : 23
}

font_italic = {'family' : 'Times New Roman',
'style' : "italic",
'weight' : 'regular',
'size' : 23
}

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
})

figure_size = (8, 6)
fig, ax = plt.subplots(figsize=figure_size)
ax.spines["left"]
ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.5, labelsize=20)
ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1.5, labelsize=20)
ax.set_xlabel(r'$t/U$', font)
ax.set_ylabel(r'$J/U$', font)
frameSize = 1.5
ax.spines['left'].set_linewidth(frameSize)
ax.spines['right'].set_linewidth(frameSize)
ax.spines['top'].set_linewidth(frameSize)
ax.spines['bottom'].set_linewidth(frameSize)
ax.set_xlim(0, 0.2)
ax.set_ylim(0, 0.05)
p1 = ax.plot(np.arange(0, 0.2, 0.005), perturbation1, color="red", label="perturbation")
p2 = ax.plot(np.arange(0, 0.2, 0.005), perturbation2, "--", color="red", label="high order perturbation")
p3 = ax.plot(np.arange(0, 0.2, 0.005), exact_diagonalization, color="blue", label="exact diagonalization")
#lns = [p1, p2, p3]
ax.legend(loc='best', frameon=False, handlelength=1.2, prop=font)
fig.tight_layout()
plt.savefig("compare_superexchange.png", format='png', dpi=600)