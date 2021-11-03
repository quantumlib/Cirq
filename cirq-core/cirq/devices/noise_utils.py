from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple, Union
import numpy as np
import warnings

if TYPE_CHECKING:
    import cirq


# Tag for gates to which noise must be applied.
PHYSICAL_GATE_TAG = 'physical_gate'


@dataclass(frozen=True)
class OpIdentifier:
    """Identifies an operation by gate and (optionally) target qubits."""
    gate: type
    qubits: List['cirq.Qid']

    def __init__(self, gate, *qubits):
        object.__setattr__(self, 'gate', gate)
        object.__setattr__(self, 'qubits', qubits)

    def gate_only(self):
        return OpIdentifier(self.gate)

    def __str__(self):
        if self.qubits:
            return f'{self.gate}{self.qubits}'
        return f'{self.gate}'

    def __repr__(self) -> str:
        return f'OpIdentifier({self.gate.__qualname__}, *{self.qubits})'


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


def decoherence_pauli_error(T1_ns: float, Tphi_ns: float, gate_time_ns: float) -> float:
    """The component of Pauli error caused by decoherence.

    Args:
        T1_ns: T1 time in nanoseconds.
        Tphi_ns: Tphi time in nanoseconds.
        gate_time_ns: Duration in nanoseconds of the gate affected by this error.
    """
    Gamma2 = (1 / (2 * T1_ns)) + 1 / Tphi_ns

    exp1 = np.exp(-gate_time_ns / T1_ns)
    exp2 = np.exp(-gate_time_ns * Gamma2)
    px = 0.25 * (1 - exp1)
    py = px
    pz = 0.5 * (1 - exp2) - px
    return px + py + pz


def unitary_entanglement_fidelity(U_actual: np.ndarray, U_ideal: np.ndarray) -> np.ndarray:
    """Entanglement fidelity between two unitaries.

    For unitary matrices, this is related to average unitary fidelity F by:

        :math:`F = \frac{F_e d + 1}{d + 1}`

    where d is the matrix dimension.

    Args:
        U_actual : Matrix whose fidelity to U_ideal will be computed. This may
            be a non-unitary matrix, i.e. the projection of a larger unitary
            matrix into the computational subspace.
        U_ideal : Unitary matrix to which U_actual will be compared.

    Both arguments may be vectorized, in that their shapes may be of the form
    (...,M,M) (as long as both shapes can be broadcast together).

    Returns:
        The entanglement fidelity between the two unitaries. For inputs with
        shape (...,M,M), the output has shape (...).
    """

    def shapes_broadcastable(shape_0: Tuple[int, ...], shape_1: Tuple[int, ...]) -> bool:
        return all((m == n) or (m == 1) or (n == 1) for m, n in zip(shape_0[::-1], shape_1[::-1]))

    U_actual = np.asarray(U_actual)
    U_ideal = np.asarray(U_ideal)
    if not shapes_broadcastable(U_actual.shape, U_ideal.shape):
        raise ValueError('Input arrays do not have matching shapes.')
    if U_actual.shape[-1] != U_actual.shape[-2]:
        raise ValueError("Inputs' trailing dimensions must be equal (square).")

    dim = U_ideal.shape[-1]

    prod_trace = np.einsum('...ba,...ba->...', U_actual.conj(), U_ideal)

    return np.real((np.abs(prod_trace)) / dim) ** 2
