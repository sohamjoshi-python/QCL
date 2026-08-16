import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pennylane as qml
import numpy as onp
import pennylane.numpy as np

import models.qcl as qcl


def load_xy_n4():
    data = onp.load("dataset/xy/data/dataset_xy_n4.npz")
    X_train, X_test = data["X_train"], data["X_test"]
    y_tr_raw, y_te_raw = data["y_train_entropy"], data["y_test_entropy"]
    y_min, y_max = y_tr_raw.min(), y_tr_raw.max()
    y_train = 2 * (y_tr_raw - y_min) / (y_max - y_min) - 1
    y_test = 2 * (y_te_raw - y_min) / (y_max - y_min) - 1
    return (np.array(X_train, requires_grad=False),
            np.array(X_test, requires_grad=False),
            np.array(y_train, requires_grad=False),
            np.array(y_test, requires_grad=False))


def train_qcl_lr(X_train, y_train, X_test, y_test, lr,
                 n_qubits=3, n_layers=6, batch_size=64, n_iters=400, seed=0):
    circuit, n_params = qcl.build_qcl(n_qubits, n_layers)
    rng = onp.random.default_rng(seed)
    params = np.array(rng.uniform(-0.1, 0.1, size=n_params), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=lr)

    for step in range(n_iters):
        idx = rng.integers(0, len(X_train), size=batch_size)
        Xb, yb = X_train[idx], y_train[idx]

        def cost(p):
            preds = np.array([circuit(p, x) for x in Xb])
            return np.mean((preds - yb) ** 2)

        params = opt.step(cost, params)

        if step % 100 == 0:
            preds = np.array([circuit(params, x) for x in X_test])
            nmse = float(np.mean((preds - y_test) ** 2)) / float(np.var(y_test))
            print(f"    lr={lr:<6} step {step}/{n_iters}: test nMSE = {nmse:.6f}")

    preds_te = np.array([circuit(params, x) for x in X_test])
    test_nmse = float(np.mean((preds_te - y_test) ** 2)) / float(np.var(y_test))
    preds_tr = np.array([circuit(params, x) for x in X_train])
    train_nmse = float(np.mean((preds_tr - y_train) ** 2)) / float(np.var(y_train))
    return train_nmse, test_nmse


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_xy_n4()
    print(f"Static XY n=4: {len(X_train)} train, {len(X_test)} test, "
          f"target=entanglement entropy (QCL 3 qubits x 6 layers, seed=0)\n")

    results = []
    for lr in [0.001, 0.01, 0.1]:
        print(f"--- learning rate = {lr} ---")
        tr, te = train_qcl_lr(X_train, y_train, X_test, y_test, lr=lr)
        results.append((lr, tr, te))
        print(f"  final: train nMSE = {tr:.6f}, test nMSE = {te:.6f}\n")

    print("=" * 48)
    print(f"{'lr':>8} | {'train nMSE':>12} | {'test nMSE':>12}")
    print("-" * 48)
    for lr, tr, te in results:
        print(f"{lr:>8} | {tr:>12.6f} | {te:>12.6f}")
    best = min(results, key=lambda r: r[2])
    print("=" * 48)
    print(f"Best (lowest test nMSE): lr={best[0]}  (test nMSE={best[2]:.6f})")
