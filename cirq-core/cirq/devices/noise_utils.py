# Copyright 2021 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence, Tuple
import warnings
import numpy as np

if TYPE_CHECKING:
    import cirq


# Tag for gates to which noise must be applied.
PHYSICAL_GATE_TAG = 'physical_gate'


@dataclass(frozen=True)
class OpIdentifier:
    """Identifies an operation by gate and (optionally) target qubits."""

    gate_type: type
    qubits: Sequence['cirq.Qid']

    def __init__(self, gate_type: type, *qubits: 'cirq.Qid'):
        object.__setattr__(self, 'gate_type', gate_type)
        object.__setattr__(self, 'qubits', qubits)

    def swapped(self):
        return OpIdentifier(self.gate_type, *self.qubits[::-1])

    def __str__(self):
        return f'{self.gate_type}{self.qubits}'

    def __repr__(self) -> str:
        fullname = f'{self.gate_type.__module__}.{self.gate_type.__qualname__}'
        qubits = ', '.join(map(repr, self.qubits))
        return f'cirq.devices.noise_utils.OpIdentifier({fullname}, {qubits})'


# TODO: expose all from top-level cirq?
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
