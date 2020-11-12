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
from cirq.qis.states import (infer_qid_shape, quantum_state,
                             validate_density_matrix,
                             validate_normalized_state_vector)

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
    # Two ints
    if isinstance(state1, int) and isinstance(state2, int):
        if validate and qid_shape is not None:
            dim = np.prod(qid_shape)
            if state1 >= dim:
                raise ValueError(
                    'Invalid state for given qid shape: '
                    'The maximum computational basis state for qid shape '
                    f'{qid_shape} is {dim - 1}, but {state1} was given.')
            if state2 >= dim:
                raise ValueError(
                    'Invalid state for given qid shape: '
                    'The maximum computational basis state for qid shape '
                    f'{qid_shape} is {dim - 1}, but {state2} was given.')
        return float(state1 == state2)

    # Two ProductStates
    if isinstance(state1, value.ProductState) and isinstance(
            state2, value.ProductState):
        if len(state1) != len(state2):
            raise ValueError('Mismatched number of qubits in product states: '
                             f'{len(state1)} and {len(state2)}.')
        if validate and qid_shape is not None and qid_shape != (
                2,) * len(state1):
            raise ValueError('Invalid state for given qid shape: '
                             f'Specified shape {qid_shape} but product state '
                             f'has shape {(2,) * len(state1)}.')
        prod = 1.0
        for q, s1 in state1:
            s2 = state2[q]
            prod *= np.abs(np.vdot(s1.state_vector(), s2.state_vector()))
        return prod**2

    # Two numpy arrays that are either state vector, state tensor, or
    # density matrix
    if (isinstance(state1, np.ndarray) and state1.dtype.kind == 'c' and
            isinstance(state2, np.ndarray) and state2.dtype.kind == 'c'):
        state1, state2 = _numpy_arrays_to_state_vectors_or_density_matrices(
            state1, state2, qid_shape=qid_shape, validate=validate)
        return _fidelity_state_vectors_or_density_matrices(state1, state2)

    # Use QuantumState machinery for the general case
    if qid_shape is None:
        qid_shape = infer_qid_shape(state1, state2)
    state1 = quantum_state(state1,
                           qid_shape=qid_shape,
                           validate=validate)
    state2 = quantum_state(state2,
                           qid_shape=qid_shape,
                           validate=validate)
    state1 = state1.density_matrix() if state1.is_density_matrix(
    ) else state1.state_vector()
    state2 = state2.density_matrix() if state2.is_density_matrix(
    ) else state2.state_vector()
    return _fidelity_state_vectors_or_density_matrices(state1, state2)


def _numpy_arrays_to_state_vectors_or_density_matrices(
        state1: np.ndarray, state2: np.ndarray,
        qid_shape: Optional[Tuple[int, ...]], validate: bool) -> Tuple[np.ndarray, np.ndarray]:
    if state1.ndim > 2 or state1.ndim == 2 and state1.shape[0] != state1.shape[
            1]:
        # State tensor, convert to state vector
        state1 = np.reshape(state1, (np.prod(state1.shape),))
    if state2.ndim > 2 or state2.ndim == 2 and state2.shape[0] != state2.shape[
            1]:
        # State tensor, convert to state vector
        state2 = np.reshape(state2, (np.prod(state2.shape),))
    if state1.ndim == 2 and state2.ndim == 2:
        if state1.shape == state2.shape:
            if qid_shape is None:
                # Ambiguous whether state tensor or density matrix
                raise ValueError(
                    'The qid shape of the given states is ambiguous.'
                    'Try specifying the qid shape explicitly or '
                    'using a wrapper function like cirq.density_matrix.')
            if state1.shape == qid_shape:
                # State tensor, convert to state vector
                state1 = np.reshape(state1, (np.prod(qid_shape),))
            if state2.shape == qid_shape:
                # State tensor, convert to state vector
                state2 = np.reshape(state2, (np.prod(qid_shape),))
        if state1.shape[0] < state2.shape[0]:
            # state1 is state tensor and state2 is density matrix.
            # Convert state1 to state vector
            state1 = np.reshape(state1, (np.prod(state1.shape),))
        else:
            # state2 is state tensor and state1 is density matrix.
            # Convert state2 to state vector
            state2 = np.reshape(state2, (np.prod(state2.shape),))
    elif state1.ndim == 2 and state2.ndim < 2:
        if np.prod(state1.shape) == np.prod(state2.shape):
            # state1 is state tensor, convert to state vector
            state1 = np.reshape(state1, (np.prod(state1.shape),))
    elif state1.ndim < 2 and state2.ndim == 2:
        if np.prod(state1.shape) == np.prod(state2.shape):
            # state2 is state tensor, convert to state vector
            state2 = np.reshape(state2, (np.prod(state2.shape),))

    if validate:
        dim1 = state1.shape[0] if state1.ndim == 2 else np.prod(state1.shape)
        dim2 = state2.shape[0] if state2.ndim == 2 else np.prod(state2.shape)
        if dim1 != dim2:
            raise ValueError('Mismatched dimensions in given states: '
                             f'{dim1} and {dim2}.')
        if qid_shape is None:
            qid_shape = (dim1,)
        else:
            expected_dim = np.prod(qid_shape)
            if dim1 != expected_dim:
                raise ValueError('Invalid state dimension for given qid shape: '
                                 f'Expected dimension {expected_dim} but '
                                 f'got dimension {dim1}.')
        if state1.ndim == 2:
            validate_density_matrix(state1, qid_shape=qid_shape)
        else:
            validate_normalized_state_vector(state1, qid_shape=qid_shape)
        if state2.ndim == 2:
            validate_density_matrix(state2, qid_shape=qid_shape)
        else:
            validate_normalized_state_vector(state2, qid_shape=qid_shape)

    return state1, state2


def _fidelity_state_vectors_or_density_matrices(state1: np.ndarray,
                                                state2: np.ndarray) -> float:
    if state1.ndim == 1 and state2.ndim == 1:
        # Both state vectors
        return np.abs(np.vdot(state1, state2))**2
    elif state1.ndim == 1 and state2.ndim == 2:
        # state1 is a state vector and state2 is a density matrix
        return np.real(np.conjugate(state1) @ state2 @ state1)
    elif state1.ndim == 2 and state2.ndim == 1:
        # state1 is a density matrix and state2 is a state vector
        return np.real(np.conjugate(state2) @ state1 @ state2)
    elif state1.ndim == 2 and state2.ndim == 2:
        # Both density matrices
        state1_sqrt = _sqrt_positive_semidefinite_matrix(state1)
        eigs = scipy.linalg.eigvalsh(state1_sqrt @ state2 @ state1_sqrt)
        trace = np.sum(np.sqrt(np.abs(eigs)))
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
    return scipy.stats.entropy(np.abs(eigenvalues), base=2)
