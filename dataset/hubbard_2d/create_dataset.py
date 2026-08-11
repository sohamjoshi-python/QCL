import os
import numpy as np
import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import hubbard_model as hub


def generate_hubbard_dataset(n, n_train, n_test, param_ranges, seed=0):
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test

    t_lo, t_hi = param_ranges['t']
    U_lo, U_hi = param_ranges['U']

    # Build skeleton ONCE
    print(f"  Building skeleton for n={n}...")
    skel = hub.build_skeleton(n)
    print(f"  Sector dim = {skel['dim']}, "
          f"{len(skel['hop_rows'])} hopping entries")

    X_all = []
    y_energy_all = []
    y_docc_all = []
    n_rejected = 0
    start = time.time()

    while len(X_all) < n_total:
        t_val = rng.uniform(t_lo, t_hi)
        U_val = rng.uniform(U_lo, U_hi)

        result = hub.solve_from_skeleton(skel, t_val, U_val)
        if result is None:
            n_rejected += 1
            continue

        X_all.append([t_val, U_val])
        y_energy_all.append(result['energy_per_site'])
        y_docc_all.append(result['double_occupancy'])

        if len(X_all) % 200 == 0:
            elapsed = time.time() - start
            rate = len(X_all) / elapsed
            remaining = (n_total - len(X_all)) / rate
            print(f"  {len(X_all)}/{n_total} "
                  f"[{elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining]")

    X = np.array(X_all)
    y_energy = np.array(y_energy_all)
    y_docc = np.array(y_docc_all)

    # Rescale each input feature to (0, 2pi) using its sampling range so that
    # QCL (and the Fourier/MLP baselines) all see inputs on a common angular
    # scale spanning a full period. Labels are computed above from the physical
    # parameters, so the physics is unaffected.
    for j, (lo, hi) in enumerate([(t_lo, t_hi), (U_lo, U_hi)]):
        X[:, j] = (X[:, j] - lo) / (hi - lo) * (2 * np.pi)

    return {
        'X_train': X[:n_train],
        'X_test': X[n_train:n_train + n_test],
        'y_train_energy': y_energy[:n_train],
        'y_test_energy': y_energy[n_train:n_train + n_test],
        'y_train_docc': y_docc[:n_train],
        'y_test_docc': y_docc[n_train:n_train + n_test],
        'n_rejected': n_rejected,
    }


if __name__ == "__main__":
    param_ranges = {
        't': (0.5, 1.5),
        'U': (0.0, 8.0),
    }
    n_train = 2000
    n_test = 500

    for n in [4, 6, 8]:
        print(f"\n{'='*60}")
        print(f"Generating dataset for n={n}")
        print(f"{'='*60}")

        start = time.time()
        data = generate_hubbard_dataset(
            n=n,
            n_train=n_train,
            n_test=n_test,
            param_ranges=param_ranges,
            seed=42,
        )
        total_time = time.time() - start

        filename = f"dataset/hubbard_2d/data/dataset_hubbard_n{n}.npz"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez(filename,
                 X_train=data['X_train'],
                 X_test=data['X_test'],
                 y_train_energy=data['y_train_energy'],
                 y_test_energy=data['y_test_energy'],
                 y_train_docc=data['y_train_docc'],
                 y_test_docc=data['y_test_docc'])

        print(f"\n  Saved {filename} ({total_time:.1f}s total)")
        print(f"  {data['n_rejected']} degenerate points rejected")
        print(f"  X_train shape: {data['X_train'].shape}")
        print(f"  t range (rescaled to 0..2pi): [{data['X_train'][:,0].min():.3f}, "
              f"{data['X_train'][:,0].max():.3f}]")
        print(f"  U range (rescaled to 0..2pi): [{data['X_train'][:,1].min():.3f}, "
              f"{data['X_train'][:,1].max():.3f}]")
        print(f"  energy range: [{data['y_train_energy'].min():.3f}, "
              f"{data['y_train_energy'].max():.3f}]")
        print(f"  docc range: [{data['y_train_docc'].min():.3f}, "
              f"{data['y_train_docc'].max():.3f}]")