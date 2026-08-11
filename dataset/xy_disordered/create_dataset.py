import os
import sys
import pathlib
import time

import numpy as np

# Import the (now array-aware) XY solver from dataset/xy.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "xy"))
import xy_model


# Fixed couplings; disorder enters only through the per-site field h_i.
Jx = 1.0
Jy = 0.5
h_lo, h_hi = 0.1, 2.0
n_train = 2000
n_test = 500


def generate_xy_disordered_dataset(n, seed=42):
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test

    X_all = []
    y_entropy_all = []
    y_energy_all = []
    n_rejected = 0
    start = time.time()

    while len(X_all) < n_total:
        h_arr = rng.uniform(h_lo, h_hi, size=n)

        is_degen, _ = xy_model.check_degeneracy(Jx, Jy, h_arr, n)
        if is_degen:
            n_rejected += 1
            continue

        X_all.append(h_arr)
        y_entropy_all.append(xy_model.entanglement_entropy(Jx, Jy, h_arr, n))
        y_energy_all.append(xy_model.energy_per_site(Jx, Jy, h_arr, n))

        if len(X_all) % 200 == 0:
            elapsed = time.time() - start
            rate = len(X_all) / elapsed
            remaining = (n_total - len(X_all)) / rate
            print(f"  {len(X_all)}/{n_total} "
                  f"[{elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining]")

    X = np.array(X_all)
    y_entropy = np.array(y_entropy_all)
    y_energy = np.array(y_energy_all)

    # Split first, then rescale each input column to [0, 2pi) using the
    # train-split min/max (labels are raw; standardization happens in the
    # run script, matching repo convention).
    X_train, X_test = X[:n_train], X[n_train:n_train + n_test]
    col_min = X_train.min(axis=0)
    col_max = X_train.max(axis=0)
    X_train = (X_train - col_min) / (col_max - col_min) * (2 * np.pi)
    X_test = (X_test - col_min) / (col_max - col_min) * (2 * np.pi)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train_entropy": y_entropy[:n_train],
        "y_test_entropy": y_entropy[n_train:n_train + n_test],
        "y_train_energy": y_energy[:n_train],
        "y_test_energy": y_energy[n_train:n_train + n_test],
        "n_rejected": n_rejected,
    }


if __name__ == "__main__":
    for n in [4, 6, 8]:
        print(f"\n{'='*60}")
        print(f"Generating disordered XY dataset for n={n} (d={n})")
        print(f"{'='*60}")

        start = time.time()
        data = generate_xy_disordered_dataset(n=n, seed=42)
        total_time = time.time() - start

        filename = f"dataset/xy_disordered/data/dataset_xy_disordered_n{n}.npz"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez(filename,
                 X_train=data["X_train"],
                 X_test=data["X_test"],
                 y_train_entropy=data["y_train_entropy"],
                 y_test_entropy=data["y_test_entropy"],
                 y_train_energy=data["y_train_energy"],
                 y_test_energy=data["y_test_energy"])

        print(f"\n  Saved {filename} ({total_time:.1f}s total)")
        print(f"  {data['n_rejected']} degenerate points rejected")
        print(f"  X_train shape: {data['X_train'].shape}")
        print(f"  entropy range: [{data['y_train_entropy'].min():.3f}, "
              f"{data['y_train_entropy'].max():.3f}]")
        print(f"  energy range: [{data['y_train_energy'].min():.3f}, "
              f"{data['y_train_energy'].max():.3f}]")
