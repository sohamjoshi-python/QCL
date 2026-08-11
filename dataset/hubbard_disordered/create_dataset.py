import os
import sys
import pathlib
import time

import numpy as np

# Import the Hubbard solver (with the new disordered ground-state routine).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hubbard_2d"))
import hubbard_model as hub


# Fixed hopping / interaction; disorder enters through the per-site potential.
t_hop = 1.0
U = 4.0
eps_lo, eps_hi = -2.0, 2.0
n_train = 2000
n_test = 500


def generate_hubbard_disordered_dataset(n, seed=42):
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test

    print(f"  Building skeleton for n={n}...")
    skel = hub.build_skeleton(n)
    print(f"  Sector dim = {skel['dim']}, "
          f"{len(skel['hop_rows'])} hopping entries")

    X_all = []
    y_docc_all = []
    y_energy_all = []
    n_rejected = 0
    start = time.time()

    while len(X_all) < n_total:
        eps = rng.uniform(eps_lo, eps_hi, size=n)

        result = hub.solve_disordered_from_skeleton(skel, t_hop, U, eps)
        if result is None:
            n_rejected += 1
            continue

        X_all.append(eps)
        y_docc_all.append(result["double_occupancy"])
        y_energy_all.append(result["energy_per_site"])

        if len(X_all) % 200 == 0:
            elapsed = time.time() - start
            rate = len(X_all) / elapsed
            remaining = (n_total - len(X_all)) / rate
            print(f"  {len(X_all)}/{n_total} "
                  f"[{elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining]")

    X = np.array(X_all)
    y_docc = np.array(y_docc_all)
    y_energy = np.array(y_energy_all)

    # Split first, then rescale each input column to [0, 2pi) using the
    # train-split min/max. Labels stay raw (standardized in the run script).
    X_train, X_test = X[:n_train], X[n_train:n_train + n_test]
    col_min = X_train.min(axis=0)
    col_max = X_train.max(axis=0)
    X_train = (X_train - col_min) / (col_max - col_min) * (2 * np.pi)
    X_test = (X_test - col_min) / (col_max - col_min) * (2 * np.pi)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train_docc": y_docc[:n_train],
        "y_test_docc": y_docc[n_train:n_train + n_test],
        "y_train_energy": y_energy[:n_train],
        "y_test_energy": y_energy[n_train:n_train + n_test],
        "n_rejected": n_rejected,
    }


if __name__ == "__main__":
    for n in [4, 6, 8]:
        print(f"\n{'='*60}")
        print(f"Generating disordered (Anderson) Hubbard dataset for n={n} (d={n})")
        print(f"{'='*60}")

        start = time.time()
        data = generate_hubbard_disordered_dataset(n=n, seed=42)
        total_time = time.time() - start

        filename = f"dataset/hubbard_disordered/data/dataset_hubbard_disordered_n{n}.npz"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez(filename,
                 X_train=data["X_train"],
                 X_test=data["X_test"],
                 y_train_docc=data["y_train_docc"],
                 y_test_docc=data["y_test_docc"],
                 y_train_energy=data["y_train_energy"],
                 y_test_energy=data["y_test_energy"])

        print(f"\n  Saved {filename} ({total_time:.1f}s total)")
        print(f"  {data['n_rejected']} degenerate points rejected")
        print(f"  X_train shape: {data['X_train'].shape}")
        print(f"  docc range: [{data['y_train_docc'].min():.3f}, "
              f"{data['y_train_docc'].max():.3f}]")
        print(f"  energy range: [{data['y_train_energy'].min():.3f}, "
              f"{data['y_train_energy'].max():.3f}]")
