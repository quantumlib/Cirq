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

import numpy as np


# TODO: expose all from top-level cirq?
def decay_constant_to_xeb_fidelity(decay_constant: float, num_qubits: int = 2) -> float:
    """Calculates the XEB fidelity from the depolarization decay constant.

    Args:
        decay_constant: Depolarization decay constant.
        num_qubits: Number of qubits.

    Returns:
        Calculated XEB fidelity.
    """
    N = 2**num_qubits
    return 1 - ((1 - decay_constant) * (1 - 1 / N))


def decay_constant_to_pauli_error(decay_constant: float, num_qubits: int = 1) -> float:
    """Calculates pauli error from the depolarization decay constant.

    Args:
        decay_constant: Depolarization decay constant.
        num_qubits: Number of qubits.

    Returns:
        Calculated Pauli error.
    """
    N = 2**num_qubits
    return (1 - decay_constant) * (1 - 1 / N / N)


def pauli_error_to_decay_constant(pauli_error: float, num_qubits: int = 1) -> float:
    """Calculates depolarization decay constant from pauli error.

    Args:
        pauli_error: The pauli error.
        num_qubits: Number of qubits.

    Returns:
        Calculated depolarization decay constant.
    """
    N = 2**num_qubits
    return 1 - (pauli_error / (1 - 1 / N / N))


def xeb_fidelity_to_decay_constant(xeb_fidelity: float, num_qubits: int = 2) -> float:
    """Calculates the depolarization decay constant from XEB fidelity.

    Args:
        xeb_fidelity: The XEB fidelity.
        num_qubits: Number of qubits.

    Returns:
        Calculated depolarization decay constant.
    """
    N = 2**num_qubits
    return 1 - (1 - xeb_fidelity) / (1 - 1 / N)


def pauli_error_from_t1(t_ns: float, t1_ns: float) -> float:
    """Calculates the pauli error from T1 decay constant.

    This computes error for a specific duration, `t`.

    Args:
        t_ns: The duration of the gate in ns.
        t1_ns: The T1 decay constant in ns.

    Returns:
        Calculated Pauli error resulting from T1 decay.
    """
    t2 = 2 * t1_ns
    return (1 - np.exp(-t_ns / t2)) / 2 + (1 - np.exp(-t_ns / t1_ns)) / 4


def average_error(decay_constant: float, num_qubits: int = 1) -> float:
    """Calculates the average error from the depolarization decay constant.

    Args:
        decay_constant: Depolarization decay constant.
        num_qubits: Number of qubits.

    Returns:
        Calculated average error.
    """
    N = 2**num_qubits
    return (1 - decay_constant) * (1 - 1 / N)


def decoherence_pauli_error(t1_ns: float, tphi_ns: float, gate_time_ns: float) -> float:
    """The component of Pauli error caused by decoherence on a single qubit.

    Args:
        t1_ns: T1 time in nanoseconds.
        tphi_ns: Tphi time in nanoseconds.
        gate_time_ns: Duration in nanoseconds of the gate affected by this error.

    Returns:
        Calculated Pauli error resulting from decoherence.
    """
    gamma_2 = (1 / (2 * t1_ns)) + 1 / tphi_ns

    exp1 = np.exp(-gate_time_ns / t1_ns)
    exp2 = np.exp(-gate_time_ns * gamma_2)
    px = 0.25 * (1 - exp1)
    py = px
    pz = 0.5 * (1 - exp2) - px
    return px + py + pz
