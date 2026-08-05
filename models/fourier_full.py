import numpy as np
from itertools import product

# ── 1. Enumerate frequency vectors ────────────────────────────────

def all_frequencies(d, L):
    """All integer vectors k with |k_j| <= L. Returns list of tuples."""
    return list(product(range(-L, L + 1), repeat=d))

def low_norm_frequencies(d, L, n_freqs):
    """Lowest-norm frequency vectors, up to n_freqs (excluding zero vector)."""
    all_k = all_frequencies(d, L)
    # Sort by norm, then lexicographically for ties
    all_k.sort(key=lambda k: (sum(ki**2 for ki in k), k))
    # Always include the zero vector (bias), then take n_freqs total
    return all_k[:n_freqs]


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
    }


