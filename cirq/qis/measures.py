# Copyright 2019 The Cirq Developers
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
"""Measures on and between quantum states and operations."""

from typing import List
import numpy as np
import scipy
import scipy.stats


from cirq import linalg
from cirq import protocols


def _sqrt_positive_semidefinite_matrix(mat: np.ndarray) -> np.ndarray:
    """Square root of a positive semidefinite matrix."""
    eigs, vecs = scipy.linalg.eigh(mat)
    # Zero out small negative entries
    eigs = np.maximum(eigs, np.zeros(eigs.shape, dtype=eigs.dtype))
    return vecs @ (np.sqrt(eigs) * vecs).T.conj()


def fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """Fidelity of two quantum states.

    The fidelity of two density matrices ρ and σ is defined as

        trace(sqrt(sqrt(ρ) σ sqrt(ρ)))^2.

    The given states can be state vectors or density matrices.

    Args:
        state1: The first state.
        state2: The second state.
    """
    if len(state1.shape) == 1 and len(state2.shape) == 1:
        # Both state vectors
        return np.abs(np.vdot(state1, state2))**2
    elif len(state1.shape) == 1 and len(state2.shape) == 2:
        # state1 is a state vector and state2 is a density matrix
        return np.real(np.conjugate(state1) @ state2 @ state1)
    elif len(state1.shape) == 2 and len(state2.shape) == 1:
        # state1 is a density matrix and state2 is a state vector
        return np.real(np.conjugate(state2) @ state1 @ state2)
    elif len(state1.shape) == 2 and len(state2.shape) == 2:
        # Both density matrices
        state1_sqrt = _sqrt_positive_semidefinite_matrix(state1)
        eigs = scipy.linalg.eigvalsh(state1_sqrt @ state2 @ state1_sqrt)
        # Zero out small negative entries
        eigs = np.maximum(eigs, np.zeros(eigs.shape, dtype=eigs.dtype))
        trace = np.sum(np.sqrt(eigs))
        return trace**2
    raise ValueError('The given arrays must be one- or two-dimensional. '
                     f'Got shapes {state1.shape} and {state2.shape}.')


def von_neumann_entropy(density_matrix: np.ndarray) -> float:
    """Calculates von Neumann entropy of density matrix in bits.
    Args:
        density_matrix: The density matrix.
    Returns:
        The calculated von Neumann entropy.
    """
    eigenvalues = np.linalg.eigvalsh(density_matrix)
    return scipy.stats.entropy(abs(eigenvalues), base=2)


def process_fidelity(clean_circuit, noisy_circuit, qubits) -> float:
    """Calculates the average fidelity of a noisy circuit.

    The code uses the Kraus representation for open circuits, when decomposing
    into noisy channels. The formula for process fidelity can be found at
    equation (2) of "Quantum Gate Fidelity in Terms of Choi Matrices" by
    Nathaniel Johnston and David W. Kribs which can be found at:
    https://arxiv.org/pdf/1102.0948.pdf

    Another useful reference is "A simple formula for the average gate fidelity
    of a quantum dynamical operation" by Michael A. Nielsen which can be found
    at:
    https://arxiv.org/pdf/quant-ph/0205035.pdf

    Args:
        clean_circuit: The perfect circuit (no noise, closed).
        noisy_circuit: The circuit with noise gates (open circuit).
        qubits: The list of qubits.
    Returns:
        A scalar that is the average (process) entropy
    """
    n = len(qubits)
    d = 2**n
    qubit_map = {q.with_dimension(1): i for i, q in enumerate(qubits)}

    kraus_operations = [clean_circuit.unitary().reshape([2] * (2 * n))]
    for op in noisy_circuit.all_operations():
        target_axes = [qubit_map[q.with_dimension(1)] for q in op.qubits]

        next_kraus_operations = []

        for op_kraus in protocols.channel(op, default=None):
            op_kraus_reshaped = np.conjugate(np.transpose(op_kraus)).reshape(
                [2] * (len(target_axes) * 2))
            for kraus_operation in kraus_operations:
                next_kraus_operation = linalg.targeted_left_multiply(
                    left_matrix=op_kraus_reshaped,
                    right_target=kraus_operation,
                    target_axes=target_axes)
                next_kraus_operations.append(next_kraus_operation)

        kraus_operations = next_kraus_operations

    EiT = [x.reshape(d, d) for x in kraus_operations]

    sum_traces = sum([abs(np.trace(x))**2 for x in EiT])

    return (d + sum_traces) / (d * (d + 1))
