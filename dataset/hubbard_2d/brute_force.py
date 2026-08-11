import numpy as np
from itertools import product

def build_hubbard_brute(t, U, n):
    """
    Build the full 4^n Hubbard Hamiltonian using
    single-site operators and Kronecker products.
    """
    dim = 4  # states per site: |0>, |↑>, |↓>, |↑↓>

    # Single-site operators (4x4 matrices)
    # Basis ordering: |0>, |↑>, |↓>, |↑↓>

    # Creation operators
    # c†_up: |0>→|↑>, |↓>→|↑↓>  (with fermionic sign for ↓→↑↓)
    c_dag_up = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0]
    ], dtype=complex)

    # c†_down: |0>→|↓>, |↑>→|↑↓>  (with fermionic sign: c†↓|↑> = -|↑↓>)
    c_dag_dn = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, -1, 0, 0]  # ← must be -1
     ], dtype=complex)

    c_up = c_dag_up.T.conj()
    c_dn = c_dag_dn.T.conj()

    # Number operators
    n_up = c_dag_up @ c_up
    n_dn = c_dag_dn @ c_dn

    # Jordan-Wigner sign operator for this site
    # (-1)^(n_up + n_dn): +1 for |0> and |↑↓>, -1 for |↑> and |↓>
    F = np.diag([1, -1, -1, 1]).astype(complex)

    full_dim = dim**n
    H = np.zeros((full_dim, full_dim), dtype=complex)

    # Interaction: U * n_up * n_dn at each site
    for i in range(n):
        op = n_up @ n_dn  # 4x4
        # Embed in full space: I ⊗ ... ⊗ op ⊗ ... ⊗ I
        full_op = np.eye(1)
        for j in range(n):
            full_op = np.kron(full_op, op if j == i else np.eye(dim))
        H += U * full_op

    # Hopping: -t * (c†_{i,σ} c_{j,σ} + h.c.) for each bond
    for i in range(n - 1):
        j = i + 1
        for c_dag, c_ann, label in [(c_dag_up, c_up, 'up'), (c_dag_dn, c_dn, 'dn')]:
            # c†_i c_j: create at site i, destroy at site j
            # Need Jordan-Wigner string between sites i and j

            # Build c†_i with F string on sites < i
            op_i = np.eye(1)
            for k in range(n):
                if k < i:
                    op_i = np.kron(op_i, F)
                elif k == i:
                    op_i = np.kron(op_i, c_dag)
                else:
                    op_i = np.kron(op_i, np.eye(dim))

            # Build c_j with F string on sites < j
            op_j = np.eye(1)
            for k in range(n):
                if k < j:
                    op_j = np.kron(op_j, F)
                elif k == j:
                    op_j = np.kron(op_j, c_ann)
                else:
                    op_j = np.kron(op_j, np.eye(dim))

            hop = op_i @ op_j  # c†_i c_j in full space
            H += -t * (hop + hop.conj().T)  # add hermitian conjugate

    return H

def hubbard_energy_U0(t, n):
    """Exact ground-state energy at U=0 (free fermions), half-filling."""
    single_particle = [-2*t * np.cos(np.pi * k / (n + 1))
                       for k in range(1, n + 1)]
    single_particle.sort()  # lowest first
    # Fill lowest n//2 levels, twice (once per spin)
    n_fill = n // 2
    return 2 * sum(single_particle[:n_fill])

print("Hamiltonian test:")
H = build_hubbard_brute(t=0, U=5, n=2)
print(len(H), "x", len(H))

eigenvalues = np.linalg.eigvalsh(H)
print("Eigenvalues of H:")
print(eigenvalues)
# Test at several system sizes
for n in [2, 4, 6]:
    H = build_hubbard_brute(t=1.0, U=0.0, n=n)
    E_brute = np.linalg.eigvalsh(H)[0].real
    E_exact = hubbard_energy_U0(t=1.0, n=n)
    print(f"n={n} U=0: brute={E_brute:.10f}, exact={E_exact:.10f}, "
          f"error={abs(E_brute - E_exact):.2e}")

    H2 = build_hubbard_brute(t=0.0, U=4.0, n=n)
    E_brute2 = np.linalg.eigvalsh(H2)[0].real
    print(f"n={n} t=0: brute={E_brute2:.10f}, exact=0.0, "
          f"error={abs(E_brute2):.2e}")