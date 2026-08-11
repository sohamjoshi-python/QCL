import numpy as np
import matplotlib.pyplot as plt

# ── XY brute-force Hamiltonian (reuse from your existing code) ────

def PauliX():
    return np.array([[0, 1], [1, 0]], dtype=complex)

def PauliY():
    return np.array([[0, -1j], [1j, 0]], dtype=complex)

def PauliZ():
    return np.array([[1, 0], [0, -1]], dtype=complex)

def H_brute(Jx, Jy, h, n):
    H = np.zeros((2**n, 2**n), dtype=complex)
    X, Y = PauliX(), PauliY()
    for i in range(n - 1):
        H += -Jx * np.kron(np.kron(np.eye(2**i), X),
                            np.kron(X, np.eye(2**(n-i-2))))
        H += -Jy * np.kron(np.kron(np.eye(2**i), Y),
                            np.kron(Y, np.eye(2**(n-i-2))))
    for i in range(n):
        H += -h * np.kron(np.kron(np.eye(2**i), PauliZ()),
                           np.eye(2**(n-i-1)))
    return H


# ── Time evolution ────────────────────────────────────────────────

def prepare_neel_state(n):
    """Néel state |↑↓↑↓...⟩ = |0101...⟩ in computational basis."""
    psi = np.zeros(2**n, dtype=complex)
    # |↑⟩ = |0⟩, |↓⟩ = |1⟩ in PauliZ basis
    # Néel: site 0 up, site 1 down, site 2 up, ...
    # Binary: site 0 is least significant bit
    neel_idx = sum(1 << i for i in range(1, n, 2))  # down on odd sites
    psi[neel_idx] = 1.0
    return psi

def quench_dynamics(Jx, Jy, h, n, t_values, psi0=None):
    H = H_brute(Jx, Jy, h, n)
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    if psi0 is None:
        psi0 = prepare_neel_state(n)

    coeffs = eigenvectors.conj().T @ psi0

    Sz_ops = []
    for i in range(n):
        Sz_i = np.kron(np.kron(np.eye(2**i), PauliZ()),
                        np.eye(2**(n - i - 1)))
        Sz_ops.append(Sz_i)

    results = []
    for t in t_values:
        phases = np.exp(-1j * eigenvalues * t)
        psi_t = eigenvectors @ (coeffs * phases)

        # Staggered magnetization: (-1)^i * <sigma_z_i>
        m_stag = 0.0
        for i, Sz_i in enumerate(Sz_ops):
            sign = (-1) ** i
            m_stag += sign * np.real(psi_t.conj() @ Sz_i @ psi_t)
        m_stag /= n
        results.append(m_stag)

    return np.array(results)


# ── Visualize ─────────────────────────────────────────────────────

if __name__ == "__main__":
    n = 8
    Jx, Jy, h = 1.0, 0.5, 0.8
    t_values = np.linspace(0, 20, 2000)

    print(f"Computing quench dynamics: n={n}, Jx={Jx}, Jy={Jy}, h={h}")
    print(f"Initial state: Néel |↑↓↑↓...⟩")
    print(f"Time points: {len(t_values)}")

    mz = quench_dynamics(Jx, Jy, h, n, t_values)

    print(f"Magnetization range: [{mz.min():.4f}, {mz.max():.4f}]")

    # Count dominant frequencies via FFT
    dt = t_values[1] - t_values[0]
    fft = np.fft.rfft(mz)
    freqs = np.fft.rfftfreq(len(t_values), d=dt)
    power = np.abs(fft)**2
    n_significant = np.sum(power > 0.01 * power.max())
    print(f"Number of significant frequency components: {n_significant}")

    plt.figure(figsize=(12, 4))
    plt.plot(t_values, mz, linewidth=0.5)
    plt.xlabel("Time t")
    plt.ylabel("⟨σ_z⟩(t)")
    plt.title(f"Quench dynamics: XY model n={n}, Jx={Jx}, Jy={Jy}, h={h}")
    plt.tight_layout()
    plt.savefig("quench_dynamics.png", dpi=150)
    plt.show()
    print("Saved quench_dynamics.png")