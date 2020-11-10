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

from typing import Optional, TYPE_CHECKING, Tuple

import numpy as np
import scipy
import scipy.stats
from cirq import value
from cirq.qis.states import infer_qid_shape, quantum_state

if TYPE_CHECKING:
    import cirq


def _sqrt_positive_semidefinite_matrix(mat: np.ndarray) -> np.ndarray:
    """Square root of a positive semidefinite matrix."""
    eigs, vecs = scipy.linalg.eigh(mat)
    return vecs @ (np.sqrt(np.abs(eigs)) * vecs).T.conj()


def fidelity(state1: 'cirq.QUANTUM_STATE_LIKE',
             state2: 'cirq.QUANTUM_STATE_LIKE',
             qid_shape: Optional[Tuple[int, ...]] = None,
             validate: bool = True) -> float:
    """Fidelity of two quantum states.

    The fidelity of two density matrices ρ and σ is defined as

        trace(sqrt(sqrt(ρ) σ sqrt(ρ)))^2.

    The given states can be state vectors or density matrices.

    Args:
        state1: The first state.
        state2: The second state.
        qid_shape: The qid shape of the given states.
        validate: Whether to check if the given states are valid quantum states.

    Returns:
        The fidelity.

    Raises:
        ValueError: The qid shape of the given states was not specified and
            could not be inferred.
    """
    if isinstance(state1, int) and isinstance(state2, int):
        return float(state1 == state2)

    if isinstance(state1, value.ProductState) and isinstance(
            state2, value.ProductState):
        if len(state1) != len(state2):
            raise ValueError('Mismatched number of qubits in product states: '
                             f'{len(state1)} and {len(state2)}.')
        if validate and qid_shape is not None and qid_shape != (2,) * len(state1):
            raise ValueError('Mismatched qid shape: Specified shape '
                             f'{qid_shape} but product state has shape '
                             f'{(2,) * len(state1)}.')
        return np.prod([
            np.abs(np.vdot(s1.state_vector(), s2.state_vector()))
            for s1, s2 in zip(state1, state2)
        ])**2

    if qid_shape is None:
        qid_shape = infer_qid_shape(state1, state2)

    state1 = quantum_state(state1,
                           qid_shape=qid_shape,
                           dtype=state1.dtype,
                           validate=validate)
    state2 = quantum_state(state2,
                           qid_shape=qid_shape,
                           dtype=state2.dtype,
                           validate=validate)

    return _fidelity_quantum_states(state1, state2)


def _fidelity_quantum_states(state1: 'cirq.QuantumState',
                             state2: 'cirq.QuantumState') -> float:
    if not state1.is_density_matrix() and not state2.is_density_matrix():
        state_vector_1 = state1.state_vector()
        state_vector_2 = state2.state_vector()
        return np.abs(np.vdot(state_vector_1, state_vector_2))**2
    elif not state1.is_density_matrix() and state2.is_density_matrix():
        state_vector_1 = state1.state_vector()
        density_matrix_2 = state2.density_matrix()
        return np.real(
            np.conjugate(state_vector_1) @ density_matrix_2 @ state_vector_1)
    elif state1.is_density_matrix() and not state2.is_density_matrix():
        density_matrix_1 = state1.density_matrix()
        state_vector_2 = state2.state_vector()
        return np.real(
            np.conjugate(state_vector_2) @ density_matrix_1 @ state_vector_2)
    elif state1.is_density_matrix() and state2.is_density_matrix():
        density_matrix_1 = state1.density_matrix()
        density_matrix_2 = state2.density_matrix()
        density_matrix_1_sqrt = _sqrt_positive_semidefinite_matrix(
            density_matrix_1)
        eigs = scipy.linalg.eigvalsh(
            density_matrix_1_sqrt @ density_matrix_2 @ density_matrix_1_sqrt)
        trace = np.sum(np.sqrt(np.abs(eigs)))
        return trace**2


def von_neumann_entropy(density_matrix: np.ndarray) -> float:
    """Calculates von Neumann entropy of density matrix in bits.
    Args:
        density_matrix: The density matrix.
    Returns:
        The calculated von Neumann entropy.
    """
    eigenvalues = np.linalg.eigvalsh(density_matrix)
    return scipy.stats.entropy(np.abs(eigenvalues), base=2)
