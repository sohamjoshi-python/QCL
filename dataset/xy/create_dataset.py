import numpy as np
import xy_model

ranges = {'Jx': (0.1, 1.5), 'Jy': (0.1, 1.5), 'h': (0.0, 3.0)}
n_train = 2000
n_test = 500

for n in [4, 8, 12, 16]:
    rng = np.random.default_rng(42)
    n_total = n_train + n_test

    X = np.column_stack([
        rng.uniform(lo, hi, size=n_total)
        for lo, hi in ranges.values()
    ])

    # Filter out degenerate points
    keep = []
    for i in range(n_total):
        is_degen, gap = xy_model.check_degeneracy(
            X[i, 0], X[i, 1], X[i, 2], n
        )
        if not is_degen:
            keep.append(i)
    X = X[keep]

    # Compute labels for each observable
    energies = np.array([
        xy_model.energy_per_site(X[i,0], X[i,1], X[i,2], n)
        for i in range(len(X))
    ])
    entropies = np.array([
        xy_model.entanglement_entropy(X[i,0], X[i,1], X[i,2], n)
        for i in range(len(X))
    ])

    # Rescale each input feature to (0, 2pi) using its sampling range so that
    # QCL (and the Fourier/MLP baselines) all see inputs on a common angular
    # scale spanning a full period. Labels are computed above from the physical
    # parameters, so the physics is unaffected.
    X_scaled = np.empty_like(X)
    for j, (lo, hi) in enumerate(ranges.values()):
        X_scaled[:, j] = (X[:, j] - lo) / (hi - lo) * (2 * np.pi)
    X = X_scaled

    # Split
    X_train, X_test = X[:n_train], X[n_train:n_train+n_test]
    y_train_E, y_test_E = energies[:n_train], energies[n_train:n_train+n_test]
    y_train_S, y_test_S = entropies[:n_train], entropies[n_train:n_train+n_test]

    # Save to disk
    np.savez(f'dataset/xy/data/dataset_xy_n{n}.npz',
             X_train=X_train, X_test=X_test,
             y_train_energy=y_train_E, y_test_energy=y_test_E,
             y_train_entropy=y_train_S, y_test_entropy=y_test_S)

    print(f"n={n}: {len(keep)} non-degenerate of {n_total}, "
          f"energy range [{y_train_E.min():.3f}, {y_train_E.max():.3f}], "
          f"entropy range [{y_train_S.min():.3f}, {y_train_S.max():.3f}]")