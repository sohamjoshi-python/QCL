import numpy as np
from itertools import combinations
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh


# ══════════════════════════════════════════════════════════════════
# BASIS UTILITIES
# ══════════════════════════════════════════════════════════════════

def enumerate_states(n, n_elec):
    states = []
    for sites in combinations(range(n), n_elec):
        bits = 0
        for s in sites:
            bits |= (1 << s)
        states.append(bits)
    return sorted(states)


def build_lookup(states):
    return {s: i for i, s in enumerate(states)}


def popcount_between(bits, i, j):
    if i > j:
        i, j = j, i
    mask = 0
    for k in range(i + 1, j):
        mask |= (1 << k)
    return bin(bits & mask).count('1')


def fermionic_sign(bits, i, j):
    return (-1) ** popcount_between(bits, i, j)


# ══════════════════════════════════════════════════════════════════
# SKELETON: precompute structure once, reuse for any (t, U)
# ══════════════════════════════════════════════════════════════════

def build_skeleton(n, n_up=None, n_dn=None):
    if n_up is None:
        n_up = n // 2
    if n_dn is None:
        n_dn = n // 2

    up_states = enumerate_states(n, n_up)
    dn_states = enumerate_states(n, n_dn)
    up_lookup = build_lookup(up_states)
    dn_lookup = build_lookup(dn_states)

    dim_up = len(up_states)
    dim_dn = len(dn_states)
    dim = dim_up * dim_dn

    # Diagonal: number of doubly occupied sites for each basis state, and the
    # per-site total occupation n_{i,up}+n_{i,dn} (used for on-site disorder).
    diag_docc = np.zeros(dim, dtype=float)
    occ_matrix = np.zeros((dim, n), dtype=float)
    for a, up in enumerate(up_states):
        for b, dn in enumerate(dn_states):
            idx = a * dim_dn + b
            diag_docc[idx] = bin(up & dn).count('1')
            for site in range(n):
                occ_matrix[idx, site] = ((up >> site) & 1) + ((dn >> site) & 1)

    # Off-diagonal: hopping connections (row, col, sign)
    hop_rows = []
    hop_cols = []
    hop_signs = []

    bonds = [(i, i + 1) for i in range(n - 1)]

    for site_i, site_j in bonds:
        # Spin-up hopping: site_j -> site_i
        for a, up in enumerate(up_states):
            if (up >> site_j) & 1 and not (up >> site_i) & 1:
                new_up = up ^ (1 << site_j) ^ (1 << site_i)
                a_new = up_lookup[new_up]
                sign = fermionic_sign(up, site_i, site_j)
                for b in range(dim_dn):
                    hop_rows.append(a_new * dim_dn + b)
                    hop_cols.append(a * dim_dn + b)
                    hop_signs.append(sign)

        # Spin-down hopping: site_j -> site_i
        for b, dn in enumerate(dn_states):
            if (dn >> site_j) & 1 and not (dn >> site_i) & 1:
                new_dn = dn ^ (1 << site_j) ^ (1 << site_i)
                b_new = dn_lookup[new_dn]
                sign = fermionic_sign(dn, site_i, site_j)
                for a in range(dim_up):
                    hop_rows.append(a * dim_dn + b_new)
                    hop_cols.append(a * dim_dn + b)
                    hop_signs.append(sign)

    hop_rows = np.array(hop_rows, dtype=int)
    hop_cols = np.array(hop_cols, dtype=int)
    hop_signs = np.array(hop_signs, dtype=float)

    return {
        'n': n,
        'dim': dim,
        'diag_docc': diag_docc,
        'occ_matrix': occ_matrix,
        'hop_rows': hop_rows,
        'hop_cols': hop_cols,
        'hop_signs': hop_signs,
        'up_states': up_states,
        'dn_states': dn_states,
    }


def H_from_skeleton_sparse(skel, t_hop, U):
    """Build sparse Hamiltonian from skeleton. Vectorized, no Python loop."""
    dim = skel['dim']
    rows = skel['hop_rows']
    cols = skel['hop_cols']
    signs = skel['hop_signs']

    # Diagonal: U * double_occupancy_count
    H_diag = diags(U * skel['diag_docc'], 0, shape=(dim, dim), format='csr')

    # Off-diagonal: hopping (both directions via H + H^T)
    vals = -t_hop * signs
    H_hop = csr_matrix((vals, (rows, cols)), shape=(dim, dim))
    H_hop = H_hop + H_hop.T

    return H_diag + H_hop


def solve_from_skeleton(skel, t_hop, U, degen_tol=1e-8):
    """Solve using precomputed skeleton. Uses sparse Lanczos for large sectors."""
    H = H_from_skeleton_sparse(skel, t_hop, U)
    n = skel['n']
    dim = skel['dim']

    if dim <= 500:
        evals, evecs = np.linalg.eigh(H.toarray())
    else:
        evals, evecs = eigsh(H, k=2, which='SA')
        idx = np.argsort(evals)
        evals, evecs = evals[idx], evecs[:, idx]

    gap = evals[1] - evals[0]
    if gap < degen_tol:
        return None

    E0 = evals[0]
    psi = evecs[:, 0]

    docc = np.sum(np.abs(psi) ** 2 * skel['diag_docc']) / n

    return {
        'energy_per_site': E0 / n,
        'double_occupancy': docc,
        'gap': gap,
    }


def solve_disordered_from_skeleton(skel, t_hop, U, eps, degen_tol=1e-8):
    """Ground state of the Anderson-Hubbard model with on-site disorder ``eps``.

    ``eps`` is an array of length ``n`` giving the on-site potential at each
    site. Uses the same ground-state-only strategy as ``solve_from_skeleton``
    (dense ``eigh`` for small sectors, sparse Lanczos otherwise), so the
    per-sample cost stays ~ms-100ms even at n=8.
    """
    H = H_from_skeleton_sparse(skel, t_hop, U)
    n = skel['n']
    dim = skel['dim']

    eps = np.asarray(eps, dtype=float)
    # On-site disorder: sum_i eps_i * (n_{i,up} + n_{i,dn}) on the diagonal.
    H = H + diags(skel['occ_matrix'] @ eps, 0, shape=(dim, dim), format='csr')

    if dim <= 500:
        evals, evecs = np.linalg.eigh(H.toarray())
    else:
        evals, evecs = eigsh(H, k=2, which='SA')
        idx = np.argsort(evals)
        evals, evecs = evals[idx], evecs[:, idx]

    gap = evals[1] - evals[0]
    if gap < degen_tol:
        return None

    E0 = evals[0]
    psi = evecs[:, 0]

    docc = np.sum(np.abs(psi) ** 2 * skel['diag_docc']) / n

    return {
        'energy_per_site': E0 / n,
        'double_occupancy': docc,
        'gap': gap,
    }


# ══════════════════════════════════════════════════════════════════
# ORIGINAL SECTOR SOLVER (for validation, single-call convenience)
# ══════════════════════════════════════════════════════════════════

def build_hubbard_sector(t_hop, U, n, n_up=None, n_dn=None):
    if n_up is None:
        n_up = n // 2
    if n_dn is None:
        n_dn = n // 2

    up_states = enumerate_states(n, n_up)
    dn_states = enumerate_states(n, n_dn)
    up_lookup = build_lookup(up_states)
    dn_lookup = build_lookup(dn_states)

    dim_up = len(up_states)
    dim_dn = len(dn_states)
    dim = dim_up * dim_dn

    H = lil_matrix((dim, dim), dtype=complex)

    for a, up in enumerate(up_states):
        for b, dn in enumerate(dn_states):
            idx = a * dim_dn + b
            double_occ = bin(up & dn).count('1')
            H[idx, idx] += U * double_occ

    bonds = [(i, i + 1) for i in range(n - 1)]

    for site_i, site_j in bonds:
        for a, up in enumerate(up_states):
            if (up >> site_j) & 1 and not (up >> site_i) & 1:
                new_up = up ^ (1 << site_j) ^ (1 << site_i)
                a_new = up_lookup[new_up]
                sign = fermionic_sign(up, site_i, site_j)
                for b in range(dim_dn):
                    idx_from = a * dim_dn + b
                    idx_to = a_new * dim_dn + b
                    H[idx_to, idx_from] += -t_hop * sign
                    H[idx_from, idx_to] += -t_hop * sign

        for b, dn in enumerate(dn_states):
            if (dn >> site_j) & 1 and not (dn >> site_i) & 1:
                new_dn = dn ^ (1 << site_j) ^ (1 << site_i)
                b_new = dn_lookup[new_dn]
                sign = fermionic_sign(dn, site_i, site_j)
                for a in range(dim_up):
                    idx_from = a * dim_dn + b
                    idx_to = a * dim_dn + b_new
                    H[idx_to, idx_from] += -t_hop * sign
                    H[idx_from, idx_to] += -t_hop * sign

    return H.tocsr(), up_states, dn_states


def solve_and_observe(t_hop, U, n, n_up=None, n_dn=None, degen_tol=1e-8):
    H, up_states, dn_states = build_hubbard_sector(t_hop, U, n, n_up, n_dn)
    dim = H.shape[0]

    if dim <= 500:
        evals, evecs = np.linalg.eigh(H.toarray())
    else:
        evals, evecs = eigsh(H, k=2, which='SA')
        idx = np.argsort(evals)
        evals, evecs = evals[idx], evecs[:, idx]

    gap = evals[1] - evals[0]
    if gap < degen_tol:
        return None

    E0 = evals[0].real
    psi = evecs[:, 0]

    dim_dn = len(dn_states)
    docc = 0.0
    for a, up in enumerate(up_states):
        for b, dn in enumerate(dn_states):
            idx = a * dim_dn + b
            double = bin(up & dn).count('1')
            docc += abs(psi[idx]) ** 2 * double
    docc /= n

    return {
        'energy_per_site': E0 / n,
        'double_occupancy': docc,
        'gap': gap,
    }


def energy_per_site(t_hop, U, n):
    result = solve_and_observe(t_hop, U, n)
    if result is None:
        return None
    return result['energy_per_site']


def double_occupancy_single(t_hop, U, n):
    result = solve_and_observe(t_hop, U, n)
    if result is None:
        return None
    return result['double_occupancy']


# ══════════════════════════════════════════════════════════════════
# BRUTE-FORCE SOLVER (validation only)
# ══════════════════════════════════════════════════════════════════

def build_hubbard_brute(t_hop, U, n):
    dim = 4

    c_dag_up = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0]
    ], dtype=complex)

    c_dag_dn = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, -1, 0, 0]
    ], dtype=complex)

    c_up = c_dag_up.T.conj()
    c_dn = c_dag_dn.T.conj()

    n_up_op = c_dag_up @ c_up
    n_dn_op = c_dag_dn @ c_dn

    F = np.diag([1, -1, -1, 1]).astype(complex)

    full_dim = dim ** n
    H = np.zeros((full_dim, full_dim), dtype=complex)

    for i in range(n):
        op = n_up_op @ n_dn_op
        full_op = np.eye(1)
        for j in range(n):
            full_op = np.kron(full_op, op if j == i else np.eye(dim))
        H += U * full_op

    for i in range(n - 1):
        j = i + 1
        for c_dag, c_ann in [(c_dag_up, c_up), (c_dag_dn, c_dn)]:
            op_i = np.eye(1)
            for k in range(n):
                if k < i:
                    op_i = np.kron(op_i, F)
                elif k == i:
                    op_i = np.kron(op_i, c_dag)
                else:
                    op_i = np.kron(op_i, np.eye(dim))

            op_j = np.eye(1)
            for k in range(n):
                if k < j:
                    op_j = np.kron(op_j, F)
                elif k == j:
                    op_j = np.kron(op_j, c_ann)
                else:
                    op_j = np.kron(op_j, np.eye(dim))

            hop = op_i @ op_j
            H += -t_hop * (hop + hop.conj().T)

    return H


def hubbard_energy_U0(t_hop, n):
    single_particle = [-2 * t_hop * np.cos(np.pi * k / (n + 1))
                       for k in range(1, n + 1)]
    single_particle.sort()
    n_fill = n // 2
    return 2 * sum(single_particle[:n_fill])


def count_particles_ops(n):
    dim = 4
    n_up_single = np.array([
        [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]
    ], dtype=complex)
    n_dn_single = np.array([
        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
    ], dtype=complex)

    full_dim = dim ** n
    N_up = np.zeros((full_dim, full_dim), dtype=complex)
    N_dn = np.zeros((full_dim, full_dim), dtype=complex)

    for i in range(n):
        op_up = np.eye(1)
        op_dn = np.eye(1)
        for j in range(n):
            if j == i:
                op_up = np.kron(op_up, n_up_single)
                op_dn = np.kron(op_dn, n_dn_single)
            else:
                op_up = np.kron(op_up, np.eye(dim))
                op_dn = np.kron(op_dn, np.eye(dim))
        N_up += op_up
        N_dn += op_dn

    return N_up, N_dn


def double_occ_op_brute(n):
    dim = 4
    n_up_single = np.array([
        [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]
    ], dtype=complex)
    n_dn_single = np.array([
        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
    ], dtype=complex)

    full_dim = dim ** n
    D = np.zeros((full_dim, full_dim), dtype=complex)

    for i in range(n):
        op = n_up_single @ n_dn_single
        full_op = np.eye(1)
        for j in range(n):
            full_op = np.kron(full_op, op if j == i else np.eye(dim))
        D += full_op

    return D


def brute_force_half_filling(t_hop, U, n):
    H_full = build_hubbard_brute(t_hop, U, n)
    evals, evecs = np.linalg.eigh(H_full)

    N_up_op, N_dn_op = count_particles_ops(n)
    target_up = n / 2
    target_dn = n / 2

    for i in range(len(evals)):
        psi = evecs[:, i]
        nup = np.real(psi.conj() @ N_up_op @ psi)
        ndn = np.real(psi.conj() @ N_dn_op @ psi)
        if abs(nup - target_up) < 0.01 and abs(ndn - target_dn) < 0.01:
            return evals[i].real, psi

    raise RuntimeError("No half-filling eigenstate found")


# ══════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Validation 1: Analytic limits")
    print("=" * 70)
    for n in [2, 4, 6]:
        res = solve_and_observe(1.0, 0.0, n)
        E_exact = hubbard_energy_U0(1.0, n)
        err = abs(res['energy_per_site'] * n - E_exact)
        status = "OK" if err < 1e-10 else "FAIL"
        print(f"  n={n} U=0:  sector={res['energy_per_site']*n:.10f}  "
              f"exact={E_exact:.10f}  err={err:.2e}  {status}")

        res2 = solve_and_observe(0.0, 4.0, n)
        E_sec2 = res2['energy_per_site'] * n
        status2 = "OK" if abs(E_sec2) < 1e-10 else "FAIL"
        print(f"  n={n} t=0:  sector={E_sec2:.10f}  exact=0.0           "
              f"err={abs(E_sec2):.2e}  {status2}")

    print()
    print("=" * 70)
    print("Validation 2: Sector vs brute-force energy (half-filling)")
    print("=" * 70)
    for n in [2, 4, 6]:
        for t_hop, U in [(1.0, 0.0), (1.0, 4.0), (0.5, 8.0),
                         (1.0, 2.0), (0.3, 6.0)]:
            res = solve_and_observe(t_hop, U, n)
            E_sec = res['energy_per_site'] * n
            E_brute, _ = brute_force_half_filling(t_hop, U, n)
            err = abs(E_sec - E_brute)
            status = "OK" if err < 1e-10 else "FAIL"
            print(f"  n={n} t={t_hop:.1f} U={U:.1f}: "
                  f"sector={E_sec:.10f}  brute={E_brute:.10f}  "
                  f"err={err:.2e}  {status}")

    print()
    print("=" * 70)
    print("Validation 3: Double occupancy (half-filling)")
    print("=" * 70)
    D_ops = {}
    for n in [2, 4]:
        for t_hop, U in [(1.0, 0.0), (1.0, 4.0), (0.5, 8.0), (1.0, 2.0)]:
            res = solve_and_observe(t_hop, U, n)
            docc_sec = res['double_occupancy']

            _, psi_brute = brute_force_half_filling(t_hop, U, n)
            if n not in D_ops:
                D_ops[n] = double_occ_op_brute(n)
            D = D_ops[n]
            docc_brute = np.real(psi_brute.conj() @ D @ psi_brute) / n

            err = abs(docc_sec - docc_brute)
            status = "OK" if err < 1e-10 else "FAIL"
            print(f"  n={n} t={t_hop:.1f} U={U:.1f}: "
                  f"sector={docc_sec:.10f}  brute={docc_brute:.10f}  "
                  f"err={err:.2e}  {status}")

    print()
    print("=" * 70)
    print("Validation 4: Skeleton solver vs original sector solver")
    print("=" * 70)
    for n in [2, 4, 6]:
        skel = build_skeleton(n)
        for t_hop, U in [(1.0, 0.0), (1.0, 4.0), (0.5, 8.0),
                         (1.0, 2.0), (0.3, 6.0)]:
            res_orig = solve_and_observe(t_hop, U, n)
            res_skel = solve_from_skeleton(skel, t_hop, U)

            if res_orig is None and res_skel is None:
                print(f"  n={n} t={t_hop:.1f} U={U:.1f}: both degenerate  OK")
                continue

            err_E = abs(res_orig['energy_per_site'] - res_skel['energy_per_site'])
            err_D = abs(res_orig['double_occupancy'] - res_skel['double_occupancy'])
            status = "OK" if err_E < 1e-10 and err_D < 1e-10 else "FAIL"
            print(f"  n={n} t={t_hop:.1f} U={U:.1f}: "
                  f"E_err={err_E:.2e}  D_err={err_D:.2e}  {status}")