import numpy as np
from sklearn.neural_network import MLPRegressor


def train_and_eval_mlp(X_train, X_test, y_train, y_test,
                       hidden_layer_sizes=(32, 32), max_iter=500,
                       random_state=42):
    """Train an sklearn MLPRegressor and return metrics.

    Inputs must be plain NumPy arrays (not pennylane.numpy).
    Returns a dict with model, train_mse, test_mse, train_nmse, test_nmse, param_count.
    """
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
    )

    mlp.fit(X_train, y_train)

    preds_test = mlp.predict(X_test)
    test_mse = float(np.mean((preds_test - y_test) ** 2))
    y_test_var = float(np.var(y_test))
    test_nmse = test_mse / y_test_var if y_test_var != 0 else float("inf")

    preds_train = mlp.predict(X_train)
    train_mse = float(np.mean((preds_train - y_train) ** 2))
    train_nmse = train_mse / float(np.var(y_train)) if float(np.var(y_train)) != 0 else float("inf")

    param_count = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)

    return {
        "model": mlp,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_nmse": train_nmse,
        "test_nmse": test_nmse,
        "param_count": param_count,
    }


if __name__ == "__main__":
    # Convenience: run standalone using the same dataset as other scripts
    n = 4
    data = np.load(f"dataset/xy/data/dataset_xy_n{n}.npz")
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train_raw = data["y_train_entropy"]
    y_test_raw = data["y_test_entropy"]

    y_min = y_train_raw.min()
    y_max = y_train_raw.max()
    y_train = 2 * (y_train_raw - y_min) / (y_max - y_min) - 1
    y_test = 2 * (y_test_raw - y_min) / (y_max - y_min) - 1

    print(f"Loaded n={n}: {len(X_train)} train, {len(X_test)} test")
    res = train_and_eval_mlp(X_train, X_test, y_train, y_test)
    print(f"\nMLP Results (n={n}, entanglement entropy):")
    print(f"  Parameters:      {res['param_count']}")
    print(f"  Train nMSE:      {res['train_nmse']:.4f}")
    print(f"  Test nMSE:       {res['test_nmse']:.4f}")