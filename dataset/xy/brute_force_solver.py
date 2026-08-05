import numpy as np
import math


def PauliX():
    return np.array([[0, 1], [1, 0]], dtype=complex)


def PauliY():
    return np.array([[0, -1j], [1j, 0]], dtype=complex)


def PauliZ():
    return np.array([[1, 0], [0, -1]], dtype=complex)


def Hamiltonian(Jx, Jy, h, n):
    """
    Construct the Hamiltonian for the XY model with transverse field.

    Parameters:
    Jx (float): Coupling constant in the x direction.
    Jy (float): Coupling constant in the y direction.
    h (float): Transverse field strength.
    n (int): Number of spins in the system.
    """

    H = np.zeros((2**n, 2**n), dtype=complex)

    # Construct the Hamiltonian
    X = PauliX()
    Y = PauliY()

    for i in range(n - 1):
        first_term = np.kron(np.kron(np.eye(2**i), X), np.kron(X, np.eye(2**(n - i - 2))))
        H += -Jx * first_term

        second_term = np.kron(np.kron(np.eye(2**i), Y), np.kron(Y, np.eye(2**(n - i - 2))))
        H += -Jy * second_term

    for i in range(n):
        third_term = np.kron(np.kron(np.eye(2**i), PauliZ()), np.eye(2**(n - i - 1)))
        H += -h * third_term

    return H



def ground_state(Jx, Jy, h, n):
    H = Hamiltonian(Jx, Jy, h, n)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return eigenvalues[0].real, eigenvectors[:, 0]

def energy_per_site(Jx, Jy, h, n):
    E0, _ = ground_state(Jx, Jy, h, n)
    return E0 / n

def magnetization(Jx, Jy, h, n):
    _, psi = ground_state(Jx, Jy, h, n)
    mz = 0.0
    for i in range(n):
        # Build sigma_z on site i: I ⊗ ... ⊗ sigma_z ⊗ ... ⊗ I
        Sz_i = np.kron(np.kron(np.eye(2**i), PauliZ()),
                       np.eye(2**(n - i - 1)))
        mz += np.real(psi.conj() @ Sz_i @ psi)
    return mz / n

def entanglement_entropy(Jx, Jy, h, n):
    _, psi = ground_state(Jx, Jy, h, n)
    n_A = n // 2
    # Reshape the 2^n vector into a (2^n_A) x (2^n_B) matrix
    # Rows = subsystem A (left half), Columns = subsystem B (right half)
    psi_matrix = psi.reshape(2**n_A, 2**(n - n_A))
    # Reduced density matrix: rho_A = Tr_B(|psi><psi|) = Psi @ Psi^dag
    rho_A = psi_matrix @ psi_matrix.conj().T
    # Eigenvalues of rho_A are the Schmidt coefficients squared
    lambdas = np.linalg.eigvalsh(rho_A)
    # Filter out zeros (they contribute 0 * log(0) = 0)
    lambdas = lambdas[lambdas > 1e-14]
    # Von Neumann entropy: S = -Tr(rho log rho) = -sum(lambda * log(lambda))
    return -np.sum(lambdas * np.log(lambdas)).real

    
    


