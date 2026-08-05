import pennylane as qml
import numpy as onp
import pennylane.numpy as np

def build_qcl(n_qubits, n_layers):
    """Build a QCL circuit. Returns (circuit_fn, n_params)."""
    n_params = 3 * n_qubits * n_layers  # 3 rotations per qubit per layer
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="backprop")
    def circuit(params, x):
        # params: shape (n_layers, n_qubits, 3) — trainable angles
        # x: shape (n_qubits,) — input features
        params = params.reshape(n_layers, n_qubits, 3)

        for layer in range(n_layers):
            # 1. Entangle — ring CNOT
            if n_qubits >= 2:
                for w in range(n_qubits - 1):
                    qml.CNOT(wires=[w, w + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])  # close the ring

            # 2. Trainable rotations
            for w in range(n_qubits):
                qml.RX(params[layer, w, 0], wires=w)
                qml.RY(params[layer, w, 1], wires=w)
                qml.RZ(params[layer, w, 2], wires=w)

            # 3. Data encoding (re-uploading)
            for w in range(n_qubits):
                if w < len(x):
                    qml.RX(x[w], wires=w)

        return qml.expval(qml.PauliZ(0))

    return circuit, n_params


def train_qcl(circuit, n_params, X_train, y_train,
              n_iters=400, lr=0.1, seed=0):
    rng = onp.random.default_rng(seed)
    params = rng.uniform(-0.1, 0.1, size=n_params)
    params = np.array(params, requires_grad=True)

    opt = qml.AdamOptimizer(stepsize=lr)

    for step in range(n_iters):
        def cost(p):
            predictions = np.array([circuit(p, x) for x in X_train])
            return np.mean((predictions - y_train) ** 2)

        params = opt.step(cost, params)

        if step % 50 == 0:
            loss = cost(params)
            print(f"  step {step}: loss = {loss:.6f}")

    # Final predictions
    predictions = np.array([circuit(params, x) for x in X_train])
    return params, predictions