import csv
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pennylane as qml
import numpy as onp
import pennylane.numpy as np

import models.qcl as qcl
import models.mlp as mlp_module
import models.fourier_full as fourier


def load_hubbard_dataset(n):
    data = onp.load(f"dataset/hubbard_2d/data/dataset_hubbard_n{n}.npz")
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train_raw = data["y_train_docc"]
    y_test_raw = data["y_test_docc"]

    y_min = y_train_raw.min()
    y_max = y_train_raw.max()
    y_train = 2 * (y_train_raw - y_min) / (y_max - y_min) - 1
    y_test = 2 * (y_test_raw - y_min) / (y_max - y_min) - 1

    X_train_q = np.array(X_train, requires_grad=False)
    X_test_q = np.array(X_test, requires_grad=False)
    y_train_q = np.array(y_train, requires_grad=False)
    y_test_q = np.array(y_test, requires_grad=False)

    return {
        "X_train_q": X_train_q,
        "X_test_q": X_test_q,
        "y_train_q": y_train_q,
        "y_test_q": y_test_q,
        "X_train_np": X_train,
        "X_test_np": X_test,
        "y_train_np": y_train,
        "y_test_np": y_test,
        "y_min": y_min,
        "y_max": y_max,
    }


def train_qcl_hubbard(X_train, y_train, X_test, y_test,
                       n_layers=6, batch_size=64, n_iters=400, seed=42):
    n_qubits = 2  # d = 2 for Hubbard (t, U)
    circuit, n_params = qcl.build_qcl(n_qubits, n_layers)
    rng = onp.random.default_rng(seed)
    params = np.array(rng.uniform(-0.1, 0.1, size=n_params), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.1)

    for step in range(n_iters):
        idx = rng.integers(0, len(X_train), size=batch_size)
        X_batch = X_train[idx]
        y_batch = y_train[idx]

        def cost(p):
            preds = np.array([circuit(p, x) for x in X_batch])
            return np.mean((preds - y_batch) ** 2)

        params = opt.step(cost, params)

        if step % 50 == 0:
            full_preds = np.array([circuit(params, x) for x in X_train])
            full_mse = float(np.mean((full_preds - y_train) ** 2))
            full_nmse = full_mse / float(np.var(y_train))
            print(f"  step {step}/{n_iters}: train nMSE = {full_nmse:.6f}")

    preds_test = np.array([circuit(params, x) for x in X_test])
    test_mse = float(np.mean((preds_test - y_test) ** 2))
    y_test_var = float(np.var(y_test))
    test_nmse = test_mse / y_test_var

    preds_train = np.array([circuit(params, x) for x in X_train])
    train_mse = float(np.mean((preds_train - y_train) ** 2))
    train_nmse = train_mse / float(np.var(y_train))

    return {
        "model": "QCL",
        "n_layers": n_layers,
        "n_params": n_params,
        "train_mse": train_mse,
        "train_nmse": train_nmse,
        "test_mse": test_mse,
        "test_nmse": test_nmse,
    }


def run_fourier_baselines(X_train_np, y_train_np, X_test_np, y_test_np,
                           d, qcl_params, L=6):
    results = []

    # Fourier full (L=6): at d=2, this is (2*6+1)^2 = 169 frequencies
    # = 337 design columns, well within 2000 training points — TRACTABLE
    freqs_full = fourier.all_frequencies(d, L)
    res_full = fourier.fourier_fit_and_eval(
        X_train_np, y_train_np, X_test_np, y_test_np,
        freqs_full, label="Full"
    )
    if res_full:
        results.append({
            "model": "Fourier full",
            "L": L,
            "freq_count": len(freqs_full),
            "n_params": res_full["n_params"],
            "train_mse": res_full["train_nmse"] * float(onp.var(y_train_np)),
            "train_nmse": res_full["train_nmse"],
            "test_mse": res_full["test_nmse"] * float(onp.var(y_test_np)),
            "test_nmse": res_full["test_nmse"],
        })

    # Fourier parameter-matched
    n_freqs_pm = qcl_params // 2
    freqs_pm = fourier.low_norm_frequencies(d, L, n_freqs_pm)
    res_pm = fourier.fourier_fit_and_eval(
        X_train_np, y_train_np, X_test_np, y_test_np,
        freqs_pm, label="Param-matched"
    )
    if res_pm:
        results.append({
            "model": "Fourier param-matched",
            "L": L,
            "freq_count": len(freqs_pm),
            "n_params": res_pm["n_params"],
            "train_mse": res_pm["train_nmse"] * float(onp.var(y_train_np)),
            "train_nmse": res_pm["train_nmse"],
            "test_mse": res_pm["test_nmse"] * float(onp.var(y_test_np)),
            "test_nmse": res_pm["test_nmse"],
        })

    return results


def run_experiment(output_csv, n_values=None):
    if n_values is None:
        n_values = [4, 6, 8]

    output_path = pathlib.Path(__file__).resolve().parent / output_csv
    print(f"Writing experiment CSV to: {output_path}")

    fieldnames = [
        "n",
        "seed",
        "model",
        "n_layers",
        "n_params",
        "freq_count",
        "L",
        "train_mse",
        "train_nmse",
        "test_mse",
        "test_nmse",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        seeds = [0, 1, 2]
        for n in n_values:
            print(f"\n{'='*60}")
            print(f"Running Hubbard experiment for n={n}")
            print(f"{'='*60}")
            data = load_hubbard_dataset(n)
            print(f"Loaded n={n}: {data['X_train_q'].shape[0]} train, "
                  f"{data['X_test_q'].shape[0]} test")
            print(f"Observable: double occupancy")
            print(f"Input dimension d=2 (t, U)")

            # QCL with multiple seeds
            for seed in seeds:
                print(f"\n--- QCL seed={seed} ---")
                qcl_res = train_qcl_hubbard(
                    data["X_train_q"], data["y_train_q"],
                    data["X_test_q"], data["y_test_q"],
                    n_layers=6,
                    batch_size=64,
                    n_iters=400,
                    seed=seed,
                )
                qcl_res["n"] = n
                qcl_res["seed"] = seed
                qcl_res["freq_count"] = ""
                qcl_res["L"] = ""
                writer.writerow(qcl_res)

            # MLP with multiple seeds
            for seed in seeds:
                print(f"\n--- MLP seed={seed} ---")
                mlp_res = mlp_module.train_and_eval_mlp(
                    data["X_train_np"], data["X_test_np"],
                    data["y_train_np"], data["y_test_np"],
                    hidden_layer_sizes=(32, 32), max_iter=500,
                    random_state=seed,
                )
                mlp_res["n"] = n
                mlp_res["seed"] = seed
                mlp_res["n_layers"] = ""
                mlp_res["n_params"] = mlp_res.pop("param_count")
                mlp_res["freq_count"] = ""
                mlp_res["L"] = ""
                writer.writerow(mlp_res)

            # Fourier baselines (deterministic, no seeds)
            d = data["X_train_np"].shape[1]  # d = 2
            qcl_params = 3 * d * 6  # 36 params
            print(f"\n--- Fourier baselines (d={d}, QCL params={qcl_params}) ---")
            fourier_results = run_fourier_baselines(
                data["X_train_np"], data["y_train_np"],
                data["X_test_np"], data["y_test_np"],
                d, qcl_params, L=6
            )
            for row in fourier_results:
                row["n"] = n
                row["seed"] = ""
                row["n_layers"] = ""
                writer.writerow(row)

            print(f"\nFinished n={n}. Results appended to {output_csv}")

    print(f"\nAll done. Results saved to {output_path}")


if __name__ == "__main__":
    run_experiment("hubbard_experiment_results.csv", n_values=[4, 6, 8])