import numpy as np
import csv
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pennylane as qml
import pennylane.numpy as pnp
import models.qcl as qcl
import models.mlp as mlp_module
import models.fourier_full as fourier

# ── 1. Generate data ──────────────────────────────────────────────

# Reuse the quench-dynamics generator defined in the dataset module.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent
                       / "dataset" / "hubbard_time_dep"))
from quench import hubbard_quench_dynamics

n_sites = 6
t_hop, U = 1.0, 8.0

# Full time series. Double occupancy oscillates at frequency ~U (period ~0.8);
# the window [0, 10] captures many of these coherent doublon oscillations.
t_all = np.linspace(0, 10, 2500)
y_all = hubbard_quench_dynamics(t_hop, U, n_sites, t_all)

# Shuffle and split
rng = np.random.default_rng(42)
idx = rng.permutation(len(t_all))
t_all = t_all[idx]
y_all = y_all[idx]

n_train = 2000
n_test = 500

X_train_raw = t_all[:n_train].reshape(-1, 1)
X_test_raw = t_all[n_train:n_train + n_test].reshape(-1, 1)
y_train_raw = y_all[:n_train]
y_test_raw = y_all[n_train:n_train + n_test]

# Rescale input to [0, 2*pi)
t_min = X_train_raw.min()
t_max = X_train_raw.max()
X_train = 2 * np.pi * (X_train_raw - t_min) / (t_max - t_min)
X_test = 2 * np.pi * (X_test_raw - t_min) / (t_max - t_min)

# Standardize labels to [-1, 1]
y_min = y_train_raw.min()
y_max = y_train_raw.max()
y_train = 2 * (y_train_raw - y_min) / (y_max - y_min) - 1
y_test = 2 * (y_test_raw - y_min) / (y_max - y_min) - 1

print(f"Dataset: {n_train} train, {n_test} test")
print(f"Input: time t (d=1), rescaled to [0, 2pi)")
print(f"Target: double occupancy")
print(f"Raw y range: [{y_train_raw.min():.4f}, {y_train_raw.max():.4f}]")


# ── 2. QCL (d=1, so n_qubits=1) ──────────────────────────────────

def run_qcl(X_tr, y_tr, X_te, y_te, n_layers, seed):
    n_qubits = 1
    circuit, n_params = qcl.build_qcl(n_qubits, n_layers)

    X_tr_q = pnp.array(X_tr, requires_grad=False)
    X_te_q = pnp.array(X_te, requires_grad=False)
    y_tr_q = pnp.array(y_tr, requires_grad=False)
    y_te_q = pnp.array(y_te, requires_grad=False)

    r = np.random.default_rng(seed)
    params = pnp.array(r.uniform(-0.1, 0.1, size=n_params), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.1)

    for step in range(400):
        batch_idx = r.integers(0, len(X_tr_q), size=64)
        X_b = X_tr_q[batch_idx]
        y_b = y_tr_q[batch_idx]

        def cost(p):
            preds = pnp.array([circuit(p, x) for x in X_b])
            return pnp.mean((preds - y_b) ** 2)

        params = opt.step(cost, params)

        if step % 100 == 0:
            fp = pnp.array([circuit(params, x) for x in X_te_q])
            step_test_mse = float(pnp.mean((fp - y_te_q) ** 2))
            step_test_nmse = step_test_mse / float(pnp.var(y_te_q))
            print(f"    step {step}: test nMSE = {step_test_nmse:.6f}")

    pred_te = pnp.array([circuit(params, x) for x in X_te_q])
    test_mse = float(pnp.mean((pred_te - y_te_q) ** 2))
    test_nmse = test_mse / float(pnp.var(y_te_q))

    pred_tr = pnp.array([circuit(params, x) for x in X_tr_q])
    train_mse = float(pnp.mean((pred_tr - y_tr_q) ** 2))
    train_nmse = train_mse / float(pnp.var(y_tr_q))

    return {
        "model": "QCL",
        "n_layers": n_layers,
        "n_params": n_params,
        "seed": seed,
        "train_nmse": train_nmse,
        "test_nmse": test_nmse,
        "pred_test": np.array(pred_te, dtype=float),
    }


# ── 3. Run everything ─────────────────────────────────────────────

results = []

# Set up incremental CSV writing: the file gets a header up front and each
# result row is appended and flushed as soon as it is produced, so partial
# results survive an interruption instead of only being written at the end.
results_dir = (pathlib.Path(__file__).resolve().parent.parent
               / "results" / "hubbard_time_dep")
results_dir.mkdir(parents=True, exist_ok=True)
results_csv = results_dir / "hubbard_time_dep_results.csv"

fieldnames = ["model", "seed", "n_layers", "n_params", "freq_count",
              "train_nmse", "test_nmse"]
csv_file = open(results_csv, "w", newline="", encoding="utf-8")
csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames,
                            restval="", extrasaction="ignore")
csv_writer.writeheader()
csv_file.flush()


def record(row):
    """Append a result row to the in-memory list and flush it to the CSV."""
    results.append(row)
    csv_writer.writerow(row)
    csv_file.flush()


# Predictions on the test set (seed 0) captured per config, for overlay plots.
L_values = [3, 6, 10, 20]
qcl_pred_by_L = {}
fourier_pred_by_L = {}
mlp_pred_test = None

# QCL at several layer depths
for L in L_values:
    for seed in [0, 1, 2]:
        print(f"\n--- QCL L={L}, seed={seed} ---")
        res = run_qcl(X_train, y_train, X_test, y_test, L, seed)
        if seed == 0:
            qcl_pred_by_L[L] = res.pop("pred_test")
        else:
            res.pop("pred_test", None)
        record(res)
        print(f"    test nMSE = {res['test_nmse']:.4f}")

# MLP
for seed in [0, 1, 2]:
    print(f"\n--- MLP seed={seed} ---")
    mlp_res = mlp_module.train_and_eval_mlp(
        X_train, X_test, y_train, y_test,
        hidden_layer_sizes=(32, 32), max_iter=500,
        random_state=seed,
    )
    if seed == 0:
        mlp_pred_test = mlp_res["model"].predict(X_test)
    record({
        "model": "MLP",
        "n_params": mlp_res["param_count"],
        "seed": seed,
        "train_nmse": mlp_res["train_nmse"],
        "test_nmse": mlp_res["test_nmse"],
    })
    print(f"    test nMSE = {mlp_res['test_nmse']:.4f}")

# Fourier baselines at multiple L values
for L in L_values:
    print(f"\n--- Fourier full L={L} ---")
    freqs = fourier.all_frequencies(1, L)
    res = fourier.fourier_fit_and_eval(
        X_train, y_train, X_test, y_test,
        freqs, label=f"Full L={L}"
    )
    if res:
        fourier_pred_by_L[L] = (
            fourier.fourier_design_matrix(X_test, freqs) @ res["coeffs"]
        )
        record({
            "model": f"Fourier full (L={L})",
            "n_params": res["n_params"],
            "freq_count": len(freqs),
            "train_nmse": res["train_nmse"],
            "test_nmse": res["test_nmse"],
        })

# ── 4. Finalize CSV ───────────────────────────────────────────────

csv_file.close()
print(f"\nSaved results CSV to {results_csv} ({len(results)} rows)")


# ── 5. Overlay plots (saved, not shown) ───────────────────────────

import matplotlib
matplotlib.use("Agg")  # save only, never open a window
import matplotlib.pyplot as plt

images_dir = results_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)

# Sort test points by time and map the rescaled input back to real time t.
order = np.argsort(X_test[:, 0])
t_real = t_min + (X_test[order, 0] / (2 * np.pi)) * (t_max - t_min)
y_actual = y_test[order]

for L in L_values:
    plt.figure(figsize=(12, 4))
    plt.plot(t_real, y_actual, color="black", linewidth=1.6,
             label="actual", zorder=5)
    if L in qcl_pred_by_L:
        plt.plot(t_real, qcl_pred_by_L[L][order], linewidth=1.0,
                 alpha=0.9, label=f"QCL (L={L})")
    if mlp_pred_test is not None:
        plt.plot(t_real, mlp_pred_test[order], linewidth=1.0,
                 alpha=0.9, label="MLP")
    if L in fourier_pred_by_L:
        plt.plot(t_real, fourier_pred_by_L[L][order], linewidth=1.0,
                 alpha=0.9, label=f"Fourier (L={L})")
    plt.xlabel("Time t")
    plt.ylabel("double occupancy (standardized to [-1, 1])")
    plt.title(f"Hubbard quench: model predictions vs actual (L={L})")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    out_path = images_dir / f"overlay_L{L}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# ── 6. Summary ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SUMMARY: Hubbard quench dynamics (double occupancy, d=1)")
print("=" * 60)
for r in results:
    name = r["model"]
    if "n_layers" in r and r.get("n_layers"):
        name += f" (L={r['n_layers']})"
    seed_str = f" seed={r['seed']}" if 'seed' in r and r['seed'] != '' else ""
    params = r.get("n_params", "?")
    print(f"  {name:30s}{seed_str:10s}  params={params:>5}  "
          f"test nMSE={r['test_nmse']:.4f}")
