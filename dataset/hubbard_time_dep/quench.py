import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

# Reuse the exact-diagonalization skeleton from the static Hubbard model.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent
                       / "hubbard_2d"))
from hubbard_model import (
    build_skeleton,
    H_from_skeleton_sparse,
    build_lookup,
)


def hubbard_quench_dynamics(t_hop, U, n, t_values):
    """
    Quench dynamics in the half-filling sector.
    Initial state: Néel = |↑,↓,↑,↓,...⟩
    Observable: double occupancy D = (1/n) sum_i <n_{i,up} n_{i,dn}>.

    The Néel state has zero doublons; hopping creates them and they oscillate
    at frequency ~U (the energy cost of a doublon) modulated by a slower
    envelope. This coherent ~U oscillation is the canonical "fast" Hubbard
    quench observable, in contrast to the staggered magnetization which decays
    smoothly as the (non-integrable) model thermalizes.
    """
    # Build sector Hamiltonian
    skel = build_skeleton(n)
    H = H_from_skeleton_sparse(skel, t_hop, U)

    # Full diagonalization of the sector (needed for time evolution)
    H_dense = H.toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(H_dense)

    # Néel initial state in the sector basis
    # |↑,↓,↑,↓,...⟩ means: up-electrons on even sites, down-electrons on odd sites
    up_states = skel['up_states']
    dn_states = skel['dn_states']
    dim_dn = len(dn_states)

    # up-bits: electrons on sites 0, 2, 4, ...
    neel_up = sum(1 << i for i in range(0, n, 2))
    # down-bits: electrons on sites 1, 3, 5, ...
    neel_dn = sum(1 << i for i in range(1, n, 2))

    up_lookup = build_lookup(up_states)
    dn_lookup = build_lookup(dn_states)

    psi0 = np.zeros(skel['dim'], dtype=complex)
    a = up_lookup[neel_up]
    b = dn_lookup[neel_dn]
    psi0[a * dim_dn + b] = 1.0

    # Project onto eigenbasis
    coeffs = eigenvectors.conj().T @ psi0

    # Double occupancy operator D = (1/n) sum_i n_{i,up} n_{i,dn}.
    # The skeleton already stores the diagonal (count of doubly-occupied sites).
    diag_docc = skel['diag_docc'] / n

    # Time evolution
    results = []
    for t in t_values:
        phases = np.exp(-1j * eigenvalues * t)
        psi_t = eigenvectors @ (coeffs * phases)
        obs = np.real(np.sum(np.abs(psi_t) ** 2 * diag_docc))
        results.append(obs)

    return np.array(results)


# ── Visualize ─────────────────────────────────────────────────────

if __name__ == "__main__":
    # Strong coupling (U=8) makes doublons oscillate at period ~2*pi/U ~ 0.8.
    # n=6 (sector dim 400) keeps the exact diagonalization fast.
    n = 6
    t_hop, U = 1.0, 8.0
    t_values = np.linspace(0, 10, 2500)

    print(f"Computing Hubbard quench dynamics: n={n}, t={t_hop}, U={U}")
    print(f"Initial state: Neel |up,dn,up,dn,...> (half filling)")
    print(f"Time points: {len(t_values)}")

    obs = hubbard_quench_dynamics(t_hop, U, n, t_values)

    print(f"Double occupancy range: [{obs.min():.4f}, {obs.max():.4f}]")

    # Count dominant frequencies via FFT
    dt = t_values[1] - t_values[0]
    fft = np.fft.rfft(obs)
    freqs = np.fft.rfftfreq(len(t_values), d=dt)
    power = np.abs(fft) ** 2
    n_significant = np.sum(power > 0.01 * power.max())
    print(f"Number of significant frequency components: {n_significant}")

    plt.figure(figsize=(12, 4))
    plt.plot(t_values, obs, linewidth=0.5)
    plt.xlabel("Time t")
    plt.ylabel("double occupancy")
    plt.title(f"Quench dynamics: Hubbard model n={n}, t={t_hop}, U={U}")
    plt.tight_layout()
    plt.savefig("quench_dynamics_hubbard.png", dpi=150)
    plt.show()
    print("Saved quench_dynamics_hubbard.png")
