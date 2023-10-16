import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True, linewidth=110)


def LowEnergy(eig_values, eig_vectors):
    low_states = 4
    eig_values = eig_values[0:low_states]
    eig_vectors = eig_vectors[:, 0:low_states]
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
    normalized_U[:, 0] = -normalized_U[:, 0]
    print(normalized_U)

    projected_Hamiltonian = np.matmul(normalized_U.conj().T, np.matmul(np.diag(eig_values), normalized_U))
    return projected_Hamiltonian, transform_U


def TwoBand(t, U):
    Hamiltonian = np.zeros((6, 6))

    # hopping term
    Hamiltonian[1, 2] = t
    Hamiltonian[2, 1] = t

    Hamiltonian[1, 3] = t
    Hamiltonian[3, 1] = t

    Hamiltonian[3, 4] = t
    Hamiltonian[4, 3] = t

    Hamiltonian[2, 4] = t
    Hamiltonian[4, 2] = t

    # Hubbard term
    Hamiltonian[1, 1] = U
    Hamiltonian[4, 4] = U


    eig_values, eig_vectors = np.linalg.eig(Hamiltonian)
    index = eig_values.argsort() # Ascending
    eig_values = eig_values[index]
    eig_vectors = eig_vectors[:,index]

    eig_values, eig_vectors = LowEnergy(eig_values, eig_vectors)

    print(eig_values)
    #print(eig_vectors)

    projected_states = np.array([[0,0,1,0,0,0], [0,0,0,1,0,0], [1,0,0,0,0,0], [0,0,0,0,0,1]]).T
    projected_Hamiltonian, normalized_U = Project(eig_values, eig_vectors, projected_states)
    print(projected_Hamiltonian)
    print(normalized_U)
    return 2*projected_Hamiltonian[0, 1]

U = 1.0
perturbation1 = []
perturbation2 = []
exact_diagonalization = []
for t in np.arange(0, -0.2, -0.005):
    perturbation1.append(4*t*t/U)
    perturbation2.append(4*t*t/U-24*t*t*t*t/U/U/U)
    exact_diagonalization.append(TwoBand(t, U))

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
ax.set_ylim(0, 0.16)
p1 = ax.plot(np.arange(0, 0.2, 0.005), perturbation1, color="red", label="perturbation")
p2 = ax.plot(np.arange(0, 0.2, 0.005), perturbation2, "--", color="red", label="high order perturbation")
p3 = ax.plot(np.arange(0, 0.2, 0.005), exact_diagonalization, color="blue", label="exact diagonalization")
#lns = [p1, p2, p3]
ax.legend(loc='best', frameon=False, handlelength=1.2, prop=font)
fig.tight_layout()
plt.savefig("compare_exchange.png", format='png', dpi=600)