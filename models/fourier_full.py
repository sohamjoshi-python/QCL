import heapq
import numpy as np
from itertools import product

# ── 1. Enumerate frequency vectors ────────────────────────────────

def all_frequencies(d, L):
    """All integer vectors k with |k_j| <= L. Returns list of tuples."""
    return list(product(range(-L, L + 1), repeat=d))

def low_norm_frequencies(d, L, n_freqs):
    """Lowest-norm frequency vectors, up to n_freqs (including the zero vector).

    Uses a best-first (Dijkstra-like) expansion from the zero vector so that we
    never materialize the full (2L+1)^d grid. Vectors are yielded in ascending
    (squared-norm, tuple) order — identical to sorting ``all_frequencies`` by
    that key — which keeps this a drop-in replacement even at large d.
    """
    if n_freqs <= 0:
        return []

    zero = (0,) * d
    # Heap entries: (squared_norm, tuple). The tuple is the secondary sort key,
    # matching the lexicographic tie-break of the original implementation.
    heap = [(0, zero)]
    visited = {zero}
    result = []

    while heap and len(result) < n_freqs:
        sq, k = heapq.heappop(heap)
        result.append(k)
        for j in range(d):
            for step in (1, -1):
                kj = k[j] + step
                if -L <= kj <= L:
                    nk = k[:j] + (kj,) + k[j + 1:]
                    if nk not in visited:
                        visited.add(nk)
                        heapq.heappush(heap, (sq - k[j] ** 2 + kj ** 2, nk))

    return result


# ── 2. Build design matrix ────────────────────────────────────────

def fourier_design_matrix(X, freq_vectors):
    """
    Build the design matrix for Fourier regression.
    Each frequency vector k gives two columns: cos(k·x) and sin(k·x).
    The zero vector gives just the bias column (cos(0)=1, sin(0)=0).

    X: shape (n_samples, d)
    freq_vectors: list of d-tuples

    Returns: design matrix, shape (n_samples, n_columns)
    """
    columns = []
    for k in freq_vectors:
        dot = X @ np.array(k, dtype=float)  # shape (n_samples,)
        columns.append(np.cos(dot))
        # Skip sin for the zero vector (it's always 0)
        if any(ki != 0 for ki in k):
            columns.append(np.sin(dot))
    return np.column_stack(columns)


# ── 3. Fit and evaluate ───────────────────────────────────────────

def fourier_fit_and_eval(X_train, y_train, X_test, y_test, freq_vectors, label=""):
    """Ridge regression in the Fourier basis."""
    Phi_train = fourier_design_matrix(X_train, freq_vectors)
    Phi_test = fourier_design_matrix(X_test, freq_vectors)

    n_cols = Phi_train.shape[1]
    n_train = len(X_train)

    print(f"  {label}: {len(freq_vectors)} frequencies -> {n_cols} design columns")

    # Check if well-determined
    if n_cols > n_train:
        print(f"  INTRACTABLE: {n_cols} columns > {n_train} training points")
        return None

    # Ridge regression: c = (Phi^T Phi + lambda I)^{-1} Phi^T y
    lam = 1e-8  # small regularization for numerical stability
    A = Phi_train.T @ Phi_train + lam * np.eye(n_cols)
    b = Phi_train.T @ y_train
    coeffs = np.linalg.solve(A, b)

    # Evaluate
    preds_train = Phi_train @ coeffs
    preds_test = Phi_test @ coeffs

    train_mse = np.mean((preds_train - y_train) ** 2)
    test_mse = np.mean((preds_test - y_test) ** 2)
    train_nmse = train_mse / np.var(y_train)
    test_nmse = test_mse / np.var(y_test)

    print(f"  Train nMSE:  {train_nmse:.4f}")
    print(f"  Test nMSE:   {test_nmse:.4f}")

    return {
        "train_nmse": train_nmse,
        "test_nmse": test_nmse,
        "n_params": n_cols,
        "coeffs": coeffs,
    }


