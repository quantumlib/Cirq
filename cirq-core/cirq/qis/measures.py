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

from cirq import protocols, value, _import
from cirq.qis.states import (
    QuantumState,
    infer_qid_shape,
    quantum_state,
    validate_density_matrix,
    validate_normalized_state_vector,
)

# We initialize these lazily, otherwise they slow global import speed.
stats = _import.LazyLoader("stats", globals(), "scipy.stats")
linalg = _import.LazyLoader("linalg", globals(), "scipy.linalg")


if TYPE_CHECKING:
    import cirq


def _sqrt_positive_semidefinite_matrix(mat: np.ndarray) -> np.ndarray:
    """Square root of a positive semidefinite matrix."""
    eigs, vecs = linalg.eigh(mat)
    return vecs @ (np.sqrt(np.abs(eigs)) * vecs).T.conj()


def _validate_int_state(state: int, qid_shape: Optional[Tuple[int, ...]]) -> None:
    if state < 0:
        raise ValueError(
            'Invalid state: A state specified as an integer must be non-negative, '
            f'but {state} was given.'
        )
    if qid_shape is not None:
        dim = np.prod(qid_shape, dtype=np.int64)
        if state >= dim:
            raise ValueError(
                'Invalid state for given qid shape: '
                'The maximum computational basis state for qid shape '
                f'{qid_shape} is {dim - 1}, but {state} was given.'
            )


def _validate_product_state(
    state: 'cirq.ProductState', qid_shape: Optional[Tuple[int, ...]]
) -> None:
    if qid_shape is not None and qid_shape != (2,) * len(state):
        raise ValueError(
            'Invalid state for given qid shape: '
            f'Specified shape {qid_shape} but product state '
            f'has shape {(2,) * len(state)}.'
        )


def fidelity(
    state1: 'cirq.QUANTUM_STATE_LIKE',
    state2: 'cirq.QUANTUM_STATE_LIKE',
    qid_shape: Optional[Tuple[int, ...]] = None,
    validate: bool = True,
    atol: float = 1e-7,
) -> float:
    r"""Fidelity of two quantum states.

    The fidelity of two density matrices ρ and σ is defined as:

    $$
        trace(\sqrt{\sqrt{\rho} \hspace{0.5em} \sigma \sqrt{\rho}})^2
    $$

    The given states can be state vectors or density matrices.

    Args:
        state1: The first state.
        state2: The second state.
        qid_shape: The qid shape of the given states.
        validate: Whether to check if the given states are valid quantum states.
        atol: Absolute numerical tolerance to use for validation.

    Returns:
        The value of the fidelity, as a float.

    Raises:
        ValueError: The qid shape of the given states was not specified and
            could not be inferred.
        ValueError: Invalid quantum state.
    """
    # Two ints
    if isinstance(state1, int) and isinstance(state2, int):
        if validate:
            _validate_int_state(state1, qid_shape)
            _validate_int_state(state2, qid_shape)
        return float(state1 == state2)

    # Two ProductStates
    if isinstance(state1, value.ProductState) and isinstance(state2, value.ProductState):
        if len(state1) != len(state2):
            raise ValueError(
                'Mismatched number of qubits in product states: '
                f'{len(state1)} and {len(state2)}.'
            )
        if validate:
            _validate_product_state(state1, qid_shape)
            _validate_product_state(state2, qid_shape)
        prod = 1.0
        for q, s1 in state1:
            s2 = state2[q]
            prod *= np.abs(np.vdot(s1.state_vector(), s2.state_vector()))
        return prod**2

    # Two numpy arrays that are either state vector, state tensor, or
    # density matrix
    if (
        isinstance(state1, np.ndarray)
        and state1.dtype.kind == 'c'
        and isinstance(state2, np.ndarray)
        and state2.dtype.kind == 'c'
    ):
        state1, state2 = _numpy_arrays_to_state_vectors_or_density_matrices(
            state1, state2, qid_shape=qid_shape, validate=validate, atol=atol
        )
        return _fidelity_state_vectors_or_density_matrices(state1, state2)

    # Use QuantumState machinery for the general case
    if qid_shape is None:
        try:
            qid_shape = infer_qid_shape(state1, state2)
        except:
            raise ValueError(
                'Failed to infer the qid shape of the given states. '
                'Please specify the qid shape explicitly using the `qid_shape` argument.'
            )
    state1 = quantum_state(state1, qid_shape=qid_shape, validate=validate, atol=atol)
    state2 = quantum_state(state2, qid_shape=qid_shape, validate=validate, atol=atol)
    state1_arr = state1.state_vector_or_density_matrix()
    state2_arr = state2.state_vector_or_density_matrix()
    return _fidelity_state_vectors_or_density_matrices(state1_arr, state2_arr)


def _numpy_arrays_to_state_vectors_or_density_matrices(
    state1: np.ndarray,
    state2: np.ndarray,
    qid_shape: Optional[Tuple[int, ...]],
    validate: bool,
    atol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if state1.ndim > 2 or (state1.ndim == 2 and state1.shape[0] != state1.shape[1]):
        # State tensor, convert to state vector
        state1 = state1.reshape(-1)
    if state2.ndim > 2 or (state2.ndim == 2 and state2.shape[0] != state2.shape[1]):
        # State tensor, convert to state vector
        state2 = state2.reshape(-1)
    if state1.ndim == 2 and state2.ndim == 2:
        # Must be square matrices
        if state1.shape == state2.shape:
            if qid_shape is None:
                # Ambiguous whether state tensor or density matrix
                raise ValueError(
                    'The qid shape of the given states is ambiguous. '
                    'Try specifying the qid shape explicitly or '
                    'using a wrapper function like cirq.density_matrix.'
                )
            if state1.shape == qid_shape:
                # State tensors, convert to state vectors
                state1 = state1.reshape(-1)
                state2 = state2.reshape(-1)
        elif state1.shape[0] < state2.shape[0]:
            # state1 is state tensor and state2 is density matrix.
            # Convert state1 to state vector
            state1 = state1.reshape(-1)
        else:  # state1.shape[0] > state2.shape[0]
            # state2 is state tensor and state1 is density matrix.
            # Convert state2 to state vector
            state2 = state2.reshape(-1)
    elif (
        state1.ndim == 2
        and state2.ndim < 2
        and np.prod(state1.shape, dtype=np.int64) == np.prod(state2.shape, dtype=np.int64)
    ):
        # state1 is state tensor, convert to state vector
        state1 = state1.reshape(-1)
    elif (
        state1.ndim < 2
        and state2.ndim == 2
        and np.prod(state1.shape, dtype=np.int64) == np.prod(state2.shape, dtype=np.int64)
    ):
        # state2 is state tensor, convert to state vector
        state2 = state2.reshape(-1)

    if validate:
        dim1: int = (
            state1.shape[0] if state1.ndim == 2 else np.prod(state1.shape, dtype=np.int64).item()
        )
        dim2: int = (
            state2.shape[0] if state2.ndim == 2 else np.prod(state2.shape, dtype=np.int64).item()
        )
        if dim1 != dim2:
            raise ValueError('Mismatched dimensions in given states: ' f'{dim1} and {dim2}.')
        if qid_shape is None:
            qid_shape = (dim1,)
        else:
            expected_dim = np.prod(qid_shape, dtype=np.int64)
            if dim1 != expected_dim:
                raise ValueError(
                    'Invalid state dimension for given qid shape: '
                    f'Expected dimension {expected_dim} but '
                    f'got dimension {dim1}.'
                )
        for state in (state1, state2):
            if state.ndim == 2:
                validate_density_matrix(state, qid_shape=qid_shape, atol=atol)
            else:
                validate_normalized_state_vector(state, qid_shape=qid_shape, atol=atol)

    return state1, state2


def _fidelity_state_vectors_or_density_matrices(state1: np.ndarray, state2: np.ndarray) -> float:
    if state1.ndim == 1 and state2.ndim == 1:
        # Both state vectors
        return np.abs(np.vdot(state1, state2)) ** 2
    elif state1.ndim == 1 and state2.ndim == 2:
        # state1 is a state vector and state2 is a density matrix
        return np.real(np.conjugate(state1) @ state2 @ state1)
    elif state1.ndim == 2 and state2.ndim == 1:
        # state1 is a density matrix and state2 is a state vector
        return np.real(np.conjugate(state2) @ state1 @ state2)
    elif state1.ndim == 2 and state2.ndim == 2:
        # Both density matrices
        state1_sqrt = _sqrt_positive_semidefinite_matrix(state1)
        eigs = linalg.eigvalsh(state1_sqrt @ state2 @ state1_sqrt)
        trace = np.sum(np.sqrt(np.abs(eigs)))
        return trace**2
    raise ValueError(
        'The given arrays must be one- or two-dimensional. '
        f'Got shapes {state1.shape} and {state2.shape}.'
    )


def von_neumann_entropy(
    state: 'cirq.QUANTUM_STATE_LIKE',
    qid_shape: Optional[Tuple[int, ...]] = None,
    validate: bool = True,
    atol: float = 1e-7,
) -> float:
    r"""Calculates the von Neumann entropy of a quantum state in bits.

    The Von Neumann entropy is defined as $ - trace( \rho ln \rho)$, for
    a density matrix $\rho$.  This gives the amount of entropy in 'ebits'
    (bits of bipartite entanglement).

    If `state` is a square matrix, it is assumed to be a density matrix rather
    than a (pure) state tensor.

    Args:
        state: The quantum state.
        qid_shape: The qid shape of the given state.
        validate: Whether to check if the given state is a valid quantum state.
        atol: Absolute numerical tolerance to use for validation.

    Returns:
        The calculated von Neumann entropy.

    Raises:
        ValueError: Invalid quantum state.

    References:
        https://en.wikipedia.org/wiki/Von_Neumann_entropy
    """
    if isinstance(state, QuantumState) and state._is_density_matrix():
        state = state.data
    if isinstance(state, np.ndarray) and state.ndim == 2 and state.shape[0] == state.shape[1]:
        if validate:
            if qid_shape is None:
                qid_shape = (state.shape[0],)
            validate_density_matrix(state, qid_shape=qid_shape, dtype=state.dtype, atol=atol)
        eigenvalues = np.linalg.eigvalsh(state)
        return stats.entropy(np.abs(eigenvalues), base=2)
    if validate:
        _ = quantum_state(state, qid_shape=qid_shape, copy=False, validate=True, atol=atol)
    return 0.0


def entanglement_fidelity(operation: 'cirq.SupportsKraus') -> float:
    r"""Returns entanglement fidelity of a given quantum channel.

    Entanglement fidelity $F_e$ of a quantum channel $E: L(H) \to L(H)$ is the overlap between
    the maximally entangled state $|\phi\rangle = \frac{1}{\sqrt{dim H}} \sum_i|i\rangle|i\rangle$
    and the state obtained by sending one half of $|\phi\rangle$ through the channel $E$, i.e.

        $$
        F_e = \langle\phi|(E \otimes I)(|\phi\rangle\langle\phi|)|\phi\rangle
        $$

    where $I: L(H) \to L(H)$ is the identity map.

    Args:
        operation: Quantum channel whose entanglement fidelity is to be computed.
    Returns:
        Entanglement fidelity of the channel represented by operation.
    """
    f = 0.0
    for k in protocols.kraus(operation):
        f += np.abs(np.trace(k)) ** 2
    n_qubits = protocols.num_qubits(operation)
    return float(f / 4**n_qubits)
