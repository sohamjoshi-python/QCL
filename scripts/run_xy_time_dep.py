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
                       / "dataset" / "xy_time_dep"))
from plot import quench_dynamics

n_sites = 8
Jx, Jy, h = 1.0, 0.5, 0.8

# Full time series
t_all = np.linspace(0, 20, 2500)
y_all = quench_dynamics(Jx, Jy, h, n_sites, t_all)

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
print(f"Target: staggered magnetization")
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
    }


# ── 3. Run everything ─────────────────────────────────────────────

results = []

# QCL at several layer depths
for L in [3, 6, 10, 20]:
    for seed in [0, 1, 2]:
        print(f"\n--- QCL L={L}, seed={seed} ---")
        res = run_qcl(X_train, y_train, X_test, y_test, L, seed)
        results.append(res)
        print(f"    test nMSE = {res['test_nmse']:.4f}")

# MLP
for seed in [0, 1, 2]:
    print(f"\n--- MLP seed={seed} ---")
    mlp_res = mlp_module.train_and_eval_mlp(
        X_train, X_test, y_train, y_test,
        hidden_layer_sizes=(32, 32), max_iter=500,
        random_state=seed,
    )
    results.append({
        "model": "MLP",
        "n_params": mlp_res["param_count"],
        "seed": seed,
        "train_nmse": mlp_res["train_nmse"],
        "test_nmse": mlp_res["test_nmse"],
    })
    print(f"    test nMSE = {mlp_res['test_nmse']:.4f}")

# Fourier baselines at multiple L values
for L in [3, 6, 10, 20]:
    print(f"\n--- Fourier full L={L} ---")
    freqs = fourier.all_frequencies(1, L)
    res = fourier.fourier_fit_and_eval(
        X_train, y_train, X_test, y_test,
        freqs, label=f"Full L={L}"
    )
    if res:
        results.append({
            "model": f"Fourier full (L={L})",
            "n_params": res["n_params"],
            "freq_count": len(freqs),
            "train_nmse": res["train_nmse"],
            "test_nmse": res["test_nmse"],
        })

# ── 4. Save results to CSV ────────────────────────────────────────

results_dir = (pathlib.Path(__file__).resolve().parent.parent
               / "results" / "xy_time_dep")
results_dir.mkdir(parents=True, exist_ok=True)
results_csv = results_dir / "xy_time_dep_results.csv"

fieldnames = ["model", "seed", "n_layers", "n_params", "freq_count",
              "train_nmse", "test_nmse"]
with open(results_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames,
                            restval="", extrasaction="ignore")
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print(f"\nSaved results CSV to {results_csv} ({len(results)} rows)")


# ── 5. Summary ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SUMMARY: Quench dynamics (staggered magnetization, d=1)")
print("=" * 60)
for r in results:
    name = r["model"]
    if "n_layers" in r and r.get("n_layers"):
        name += f" (L={r['n_layers']})"
    seed_str = f" seed={r['seed']}" if 'seed' in r and r['seed'] != '' else ""
    params = r.get("n_params", "?")
    print(f"  {name:30s}{seed_str:10s}  params={params:>5}  "
          f"test nMSE={r['test_nmse']:.4f}")