import numpy as np

def PauliX():
    return np.array([[0, 1], [1, 0]], dtype=complex)


def PauliY():
    return np.array([[0, -1j], [1j, 0]], dtype=complex)


def PauliZ():
    return np.array([[1, 0], [0, -1]], dtype=complex)


def A(Jx, Jy, h, n):
    matrix = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 2 * h
            elif i == j + 1 or j == i + 1:
                matrix[i, j] = -Jx - Jy
            else:
                matrix[i, j] = 0

    return matrix

def B(Jx, Jy, n):
    matrix = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 0
            elif i == j + 1:
                matrix[i, j] = Jx - Jy
            elif j == i + 1:
                matrix[i, j] = Jy - Jx
            else:
                matrix[i, j] = 0

    return matrix

def Hamiltonian(Jx, Jy, h, n):
    A_matrix = A(Jx, Jy, h, n)
    B_matrix = B(Jx, Jy, n)

    H = np.block([[A_matrix, B_matrix], [-B_matrix, -A_matrix]])

    return H

def energy_per_site(Jx, Jy, h, n):
    M = Hamiltonian(Jx, Jy, h, n)
    eigenvalues = np.linalg.eigvalsh(M)
    positive = eigenvalues[eigenvalues > 1e-12]
    return (-0.5 * np.sum(positive)) / n

def entanglement_entropy(Jx, Jy, h, n):
    M = Hamiltonian(Jx, Jy, h, n)
    evals, evecs = np.linalg.eigh(M)
    W_pos = evecs[:, evals > 1e-12]
    G = W_pos @ W_pos.conj().T

    n_A = n // 2
    idx = list(range(n_A)) + list(range(n, n + n_A))
    G_A = G[np.ix_(idx, idx)]

    nu = np.linalg.eigvalsh(G_A).real
    S = 0.0
    for v in nu:
        if v > 1e-14 and v < 1 - 1e-14:
            S -= v * np.log(v) + (1 - v) * np.log(1 - v)
    return S / 2.0


def magnetization(Jx, Jy, h, n):
    M = Hamiltonian(Jx, Jy, h, n)
    evals, evecs = np.linalg.eigh(M)
    W_pos = evecs[:, evals > 1e-12]
    G = W_pos @ W_pos.conj().T
    Cdd = G[n:, n:]  # <c†_i c_j> block
    
    mz = 0.0
    for i in range(n):
        mz += 1 - 2 * Cdd[i, i].real
    return mz / n

def check_degeneracy(Jx, Jy, h, n, tol=1e-8):
    M = Hamiltonian(Jx, Jy, h, n)
    evals = np.linalg.eigvalsh(M)
    gap = np.min(np.abs(evals))
    return gap < tol, gap

def generate_dataset(n, observable_fn, n_train, n_test,
                     param_ranges, seed=0):
    """
    Parameters:
        n: system size
        observable_fn: function(Jx, Jy, h, n) -> scalar
        n_train, n_test: number of points
        param_ranges: dict like {'Jx': (0.1, 2.0), 
                                  'Jy': (0.1, 2.0),
                                  'h': (0.0, 2.0)}
        seed: random seed
    """
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test

    # Sample random parameter points
    X = np.column_stack([
        rng.uniform(lo, hi, size=n_total)
        for lo, hi in param_ranges.values()
    ])

    # Compute labels
    y = np.array([
        observable_fn(X[i, 0], X[i, 1], X[i, 2], n)
        for i in range(n_total)
    ])

    # Split
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return X_train, y_train, X_test, y_test


ranges = {'Jx': (0.1, 1.5), 'Jy': (0.1, 1.5), 'h': (0.0, 3.0)}

for n in [4, 8, 12]:
    X_train, y_train, X_test, y_test = generate_dataset(
        n=n,
        observable_fn=energy_per_site,
        n_train=2000,
        n_test=500,
        param_ranges=ranges,
        seed=42,
    )
    print(f"n={n}: X_train {X_train.shape}, y range [{y_train.min():.3f}, {y_train.max():.3f}]")