import numpy as np
import warnings

# TODO: expose all from top-level cirq
def decay_constant_to_xeb_fidelity(decay_constant: float, num_qubits: int = 2) -> float:
    """Calculates the XEB fidelity from the depolarization decay constant.

    Args:
        decay_constant: Depolarization decay constant
        num_qubits: Number of qubits
    """
    N = 2 ** num_qubits
    return 1 - ((1 - decay_constant) * (1 - 1 / N))


def decay_constant_to_pauli_error(decay_constant: float, num_qubits: int = 1) -> float:
    """Calculates pauli error from the depolarization decay constant.

    Args:
        decay_constant: Depolarization decay constant
        num_qubits: Number of qubits
    """
    N = 2 ** num_qubits
    return (1 - decay_constant) * (1 - 1 / N / N)


def pauli_error_to_decay_constant(pauli_error: float, num_qubits: int = 1) -> float:
    """Calculates depolarization decay constant from pauli error.

    Args:
        pauli_error: The pauli error
        num_qubits: Number of qubits
    """
    N = 2 ** num_qubits
    return 1 - (pauli_error / (1 - 1 / N / N))


def xeb_fidelity_to_decay_constant(xeb_fidelity: float, num_qubits: int = 2) -> float:
    """Calculates the depolarization decay constant from XEB fidelity.

    Args:
        xeb_fidelity: The XEB fidelity
        num_qubits: Number of qubits
    """
    N = 2 ** num_qubits
    return 1 - (1 - xeb_fidelity) / (1 - 1 / N)


def pauli_error_from_t1(t: float, t1_ns: float) -> float:
    """Calculates the pauli error from T1 decay constant.

    This computes error for a specific duration, `t`.

    Args:
        t: The duration of the gate
        t1_ns: The T1 decay constant in ns
    """
    t2 = 2 * t1_ns
    return (1 - np.exp(-t / t2)) / 2 + (1 - np.exp(-t / t1_ns)) / 4


def pauli_error_from_depolarization(t: float, t1_ns: float, pauli_error: float = 0) -> float:
    """Calculates the amount of pauli error from depolarization.

    This computes non-T1 error for a specific duration, `t`. If pauli error
    from T1 decay is more than total pauli error, this returns zero; otherwise,
    it returns the portion of pauli error not attributable to T1 error.

    Args:
        t: The duration of the gate
        t1_ns: The T1 decay constant in ns
        pauli_error: The pauli error
    """
    t1_pauli_error = pauli_error_from_t1(t, t1_ns)
    if pauli_error >= t1_pauli_error:
        return pauli_error - t1_pauli_error
    else:
        warnings.warn("Pauli error from T1 decay is greater than total Pauli error", RuntimeWarning)
        return 0


def average_error(decay_constant: float, num_qubits: int = 1) -> float:
    """Calculates the average error from the depolarization decay constant.

    Args:
        decay_constant: Depolarization decay constant
        num_qubits: Number of qubits
    """
    N = 2 ** num_qubits
    return (1 - decay_constant) * (1 - 1 / N)
