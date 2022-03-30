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
"""Classes and methods for quantum states."""

from typing import (
    Any,
    cast,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    TYPE_CHECKING,
    Tuple,
    Union,
)
import itertools

import numpy as np

from cirq import value
from cirq._doc import document

if TYPE_CHECKING:
    import cirq
    from numpy.typing import DTypeLike

DEFAULT_COMPLEX_DTYPE = np.complex64

STATE_VECTOR_LIKE = Union[
    # Full big-endian computational basis state index.
    int,
    # Per-qudit computational basis values.
    Sequence[int],
    # Explicit state vector or state tensor.
    np.ndarray,
    Sequence[Union[int, float, complex]],
    # Product state object
    'cirq.ProductState',
]
document(STATE_VECTOR_LIKE, """An object representing a state vector.""")  # type: ignore

QUANTUM_STATE_LIKE = Union[
    # state vector
    STATE_VECTOR_LIKE,
    # density matrix
    np.ndarray,
    # quantum state object
    'cirq.QuantumState',
]
document(QUANTUM_STATE_LIKE, """An object representing a quantum state.""")  # type: ignore


class QuantumState:
    """A quantum state.

    Can be a state vector, a state tensor, or a density matrix.
    """

    def __init__(
        self,
        data: np.ndarray,
        qid_shape: Optional[Tuple[int, ...]] = None,
        *,  # Force keyword arguments
        dtype: Optional['DTypeLike'] = None,
        validate: bool = True,
        atol: float = 1e-7,
    ) -> None:
        """Initialize a quantum state object.

        Args:
            data: The data representing the quantum state.
            qid_shape: The qid shape.
            validate: Whether to check if the given data and qid shape
                represent a valid quantum state with the given dtype.
            dtype: The expected data type of the quantum state.
            atol: Absolute numerical tolerance to use for validation.

        Raises:
            ValueError: The qid shape was not specified and could not be
                inferred.
            ValueError: Invalid quantum state.
        """
        if qid_shape is None:
            qid_shape = infer_qid_shape(data)
        self._data = data
        self._qid_shape = qid_shape
        self._dim = np.prod(self.qid_shape, dtype=np.int64).item()
        if validate:
            self.validate(dtype=dtype, atol=atol)

    @property
    def data(self) -> np.ndarray:
        """The data underlying the quantum state."""
        return self._data

    @property
    def qid_shape(self) -> Tuple[int, ...]:
        """The qid shape of the quantum state."""
        return self._qid_shape

    @property
    def dtype(self) -> np.dtype:
        """The data type of the quantum state."""
        return self._data.dtype

    def state_vector(self) -> Optional[np.ndarray]:
        """Return the state vector of this state.

        A state vector stores the amplitudes of a pure state as a
        one-dimensional array.
        If the state is a density matrix, this method returns None.
        """
        if self._is_density_matrix():
            return None
        return np.reshape(self.data, (self._dim,))

    def state_tensor(self) -> Optional[np.ndarray]:
        """Return the state tensor of this state.

        A state tensor stores the amplitudes of a pure state as an array with
        shape equal to the qid shape of the state.
        If the state is a density matrix, this method returns None.
        """
        if self._is_density_matrix():
            return None
        return np.reshape(self.data, self.qid_shape)

    def density_matrix(self) -> np.ndarray:
        """Return the density matrix of this state.

        A density matrix stores the entries of a density matrix as a matrix
        (a two-dimensional array).
        """
        if not self._is_density_matrix():
            state_vector = self.state_vector()
            assert state_vector is not None, 'only None if _is_density_matrix'
            return np.outer(state_vector, np.conj(state_vector))
        return self.data

    def state_vector_or_density_matrix(self) -> np.ndarray:
        """Return the state vector or density matrix of this state.

        If the state is a denity matrix, return the density matrix. Otherwise, return the state
        vector.
        """
        state_vector = self.state_vector()
        if state_vector is not None:
            return state_vector
        return self.data

    def _is_density_matrix(self) -> bool:
        """Whether this quantum state is a density matrix."""
        return self.data.shape == (self._dim, self._dim)

    def validate(
        self, *, dtype: Optional['DTypeLike'] = None, atol=1e-7  # Force keyword arguments
    ) -> None:
        """Check if this quantum state is valid.

        Args:
            dtype: The expected data type of the quantum state.
            atol: Absolute numerical tolerance to use for validation.

        Raises:
            ValueError: Invalid quantum state.
        """
        is_state_vector = self.data.shape == (self._dim,)
        is_state_tensor = self.data.shape == self.qid_shape
        if is_state_vector or is_state_tensor:
            state_vector = self.state_vector()
            assert state_vector is not None
            validate_normalized_state_vector(
                state_vector,
                qid_shape=self.qid_shape,
                dtype=dtype,
                atol=atol,
            )
        elif self._is_density_matrix():
            validate_density_matrix(
                self.density_matrix(), qid_shape=self.qid_shape, dtype=dtype, atol=atol
            )
        else:
            raise ValueError(
                'Invalid quantum state: '
                f'Data shape of {self.data.shape} is not '
                f'compatible with qid shape of {self.qid_shape}.'
            )


def quantum_state(
    state: 'cirq.QUANTUM_STATE_LIKE',
    qid_shape: Optional[Tuple[int, ...]] = None,
    *,  # Force keyword arguments
    copy: bool = False,
    validate: bool = True,
    dtype: Optional['DTypeLike'] = None,
    atol: float = 1e-7,
) -> QuantumState:
    """Create a QuantumState object from a state-like object.

    Args:
        state: The state-like object.
        qid_shape: The qid shape.
        copy: Whether to copy the data underlying the state.
        validate: Whether to check if the given data and qid shape
            represent a valid quantum state with the given dtype.
        dtype: The desired data type.
        atol: Absolute numerical tolerance to use for validation.

    Raises:
        ValueError: Invalid quantum state.
        ValueError: The qid shape was not specified and could not be inferred.
    """
    if isinstance(state, QuantumState):
        if qid_shape is not None and state.qid_shape != qid_shape:
            raise ValueError(
                'The specified qid shape must be the same as the '
                'qid shape of the given state.\n'
                f'Specified shape: {qid_shape}\n'
                f'Shape of state: {state.qid_shape}.'
            )
        if copy or dtype and dtype != state.dtype:
            if dtype and dtype != state.dtype:
                data = state.data.astype(dtype, casting='unsafe', copy=True)
            else:
                data = state.data.copy()
            new_state = QuantumState(data, state.qid_shape)
        else:
            new_state = state
        if validate:
            new_state.validate(dtype=dtype, atol=atol)
        return new_state

    if isinstance(state, value.ProductState):
        actual_qid_shape = (2,) * len(state)
        if qid_shape is not None and qid_shape != actual_qid_shape:
            raise ValueError(
                'The specified qid shape must be the same as the '
                'qid shape of the given state.\n'
                f'Specified shape: {qid_shape}\n'
                f'Shape of state: {actual_qid_shape}.'
            )
        if dtype is None:
            dtype = DEFAULT_COMPLEX_DTYPE
        data = state.state_vector().astype(dtype, casting='unsafe', copy=False)
        qid_shape = actual_qid_shape
    elif isinstance(state, int):
        if qid_shape is None:
            raise ValueError(
                'The qid shape of the given state is ambiguous. '
                'Please specify the qid shape explicitly using '
                'the qid_shape argument.'
            )
        dim = np.prod(qid_shape, dtype=np.int64).item()
        if not 0 <= state < dim:
            raise ValueError(
                f'Computational basis state is out of range.\n'
                f'\n'
                f'state={state!r}\n'
                f'MIN_STATE=0\n'
                f'MAX_STATE=product(qid_shape)-1={dim-1}\n'
                f'qid_shape={qid_shape!r}\n'
            )
        if dtype is None:
            dtype = DEFAULT_COMPLEX_DTYPE
        data = one_hot(index=state, shape=(dim,), dtype=dtype)
    else:
        data = np.array(state, copy=False)
        if qid_shape is None:
            qid_shape = infer_qid_shape(state)
        if data.ndim == 1 and data.dtype.kind != 'c':
            if len(qid_shape) == np.prod(qid_shape, dtype=np.int64):
                raise ValueError(
                    'Because len(qid_shape) == product(qid_shape), it is '
                    'ambiguous whether the given state contains '
                    'state vector amplitudes or per-qudit computational basis '
                    'values. In this situation you are required to pass '
                    'in a state vector that is a numpy array with a complex '
                    'dtype.'
                )
            if data.shape == (len(qid_shape),):
                # array contains per-qudit computational basis values
                data = _qudit_values_to_state_tensor(
                    state_vector=data, qid_shape=qid_shape, dtype=dtype
                )
        if copy or dtype and dtype != data.dtype:
            if dtype and dtype != data.dtype:
                data = data.astype(dtype, casting='unsafe', copy=True)
            else:
                data = data.copy()
    return QuantumState(data=data, qid_shape=qid_shape, validate=validate, dtype=dtype, atol=atol)


def density_matrix(
    state: np.ndarray,
    qid_shape: Optional[Tuple[int, ...]] = None,
    *,  # Force keyword arguments
    copy: bool = False,
    validate: bool = True,
    dtype: Optional['DTypeLike'] = None,
    atol: float = 1e-7,
) -> QuantumState:
    """Create a QuantumState object from a density matrix.

    Args:
        state: The density matrix.
        qid_shape: The qid shape.
        copy: Whether to copy the density matrix.
        validate: Whether to check if the given data and qid shape
            represent a valid quantum state with the given dtype.
        dtype: The expected data type.
        atol: Absolute numerical tolerance to use for validation.

    Raises:
        ValueError: Invalid density matrix.
    """
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError('A density matrix must be a square matrix. ' f'Got shape {state.shape}.')
    dim, _ = state.shape
    if qid_shape is None:
        qid_shape = _infer_qid_shape_from_dimension(dim)
    return QuantumState(
        data=state.copy() if copy else state,
        qid_shape=qid_shape,
        dtype=dtype,
        validate=validate,
        atol=atol,
    )


def _infer_qid_shape_from_dimension(dim: int) -> Tuple[int, ...]:
    if dim != 0 and dim & dim - 1 == 0:
        # dim is a power of 2, assume qubits
        n_qubits = dim.bit_length() - 1
        return (2,) * n_qubits
    # dim is not a power of 2, assume a single qudit
    return (dim,)


_NON_INT_STATE_LIKE = Union[
    # Per-qudit computational basis values.
    Sequence[int],
    # Explicit state vector or state tensor.
    np.ndarray,
    Sequence[Union[int, float, complex]],
    # Product state object
    'cirq.ProductState',
    # Quantum state object
    'cirq.QuantumState',
]


def infer_qid_shape(*states: 'cirq.QUANTUM_STATE_LIKE') -> Tuple[int, ...]:
    """Infer the qid shape of a set of states.

    This is a heuristic to determine a qid shape compatible with all of the
    given states. It works by attempting to find the intersection of the sets
    of potential qid shapes for each given state. It may fail (raising an
    error) even if there is a unique compatible qid shape. If the dimension of
    a state vector or density matrix (but not state tensor) is a power of 2,
    then the state space is assumed to be composed of qubits; otherwise, it is
    assumed to be composed of a single qudit. If the qid shape is ambiguous,
    an error is raised.

    Args:
        *states: The states for which to infer the qid shape.

    Returns:
        The inferred qid shape.

    Raises:
        ValueError: The qid shape of the given states is ambiguous.
        ValueError: Failed to infer the qid shape of the given states.
    """
    if not states:
        raise ValueError('No states were specified.')

    integer_states: List[int] = []
    non_integer_states: List[_NON_INT_STATE_LIKE] = []
    for state in states:
        if isinstance(state, int):
            integer_states.append(state)
        else:
            non_integer_states.append(state)

    # if all states are specified as integers, use the minimal possible dimensions
    if not non_integer_states:
        return (max(integer_states) + 1,)

    # attempt to infer qid shape from non-integer states
    potential_shapes = _potential_qid_shapes(non_integer_states[0])
    for state in non_integer_states[1:]:
        potential_shapes = potential_shapes.intersection_subset(_potential_qid_shapes(state))
    qid_shape = potential_shapes.infer_qid_shape()
    if qid_shape is None:
        raise ValueError(
            'Failed to infer the qid shape of the given states. '
            'If you think there is an unambiguous result for this case, please raise '
            'an issue at https://github.com/quantumlib/Cirq.'
        )

    # check if the shape is compatible with the states specified as integers
    if integer_states and np.prod(qid_shape, dtype=np.int64) <= max(integer_states):
        raise ValueError(
            'Failed to infer the qid shape of the given states. '
            f'The given integer state {max(integer_states)} is too high for the '
            f'qid shape {qid_shape} inferred from some other given states.'
        )

    return qid_shape


def _potential_qid_shapes(state: _NON_INT_STATE_LIKE) -> '_QidShapeSet':
    """Return a set of qid shapes compatible with a given state."""
    if isinstance(state, QuantumState):
        return _QidShapeSet(explicit_qid_shapes={state.qid_shape})
    if isinstance(state, value.ProductState):
        return _QidShapeSet(explicit_qid_shapes={(2,) * len(state)})

    if isinstance(state, Sequence):
        state = np.array(state)
    if state.ndim == 1:
        (dim,) = state.shape
        min_qudit_dimensions = None
        if state.dtype.kind[0] in '?bBiu':
            # could be per-qudit computational basis values
            min_qudit_dimensions = tuple(state.astype(int, copy=False) + 1)
        return _QidShapeSet(
            unfactorized_total_dimension=dim, min_qudit_dimensions=min_qudit_dimensions
        )
    if state.ndim == 2:
        dim, _ = state.shape
        return _QidShapeSet(
            # could be a state tensor
            explicit_qid_shapes={state.shape},
            # could also be a density matrix
            unfactorized_total_dimension=dim,
        )
    # state tensor
    return _QidShapeSet(explicit_qid_shapes={state.shape})


class _QidShapeSet:
    """A potentially infinite set of possible qid shapes."""

    def __init__(
        self,
        *,
        explicit_qid_shapes: Optional[Set[Tuple[int, ...]]] = None,
        unfactorized_total_dimension: Optional[int] = None,
        min_qudit_dimensions: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Create a qid shape set.

        The set of qid shapes is represented as the union of a set of shapes
        specified explicitly in `explicit_qid_shapes`, a set of shapes
        specified implicitly in `unfactorized_total_dimension`, and a set of
        shapes specified implicitly in `min_qudit_dimensions`.

        Args:
            explicit_qid_shapes: An explicit set of qid shapes.
            unfactorized_total_dimension: A number representing the dimension
                of the Hilbert space. The associated qid shapes are those compatible
                with this dimension, i.e., those for which the product of the
                individual qudit dimensions is equal to the Hilbert space
                dimension.
            min_qudit_dimensions: A tuple of integers (n_1, ..., n_k).
                The associated qid shapes are
                {(m_1, ..., m_k) : m_i ≥ n_i for all i}.
        """
        self.explicit_qid_shapes = explicit_qid_shapes or set()
        self.unfactorized_total_dimension = unfactorized_total_dimension
        self.min_qudit_dimensions = min_qudit_dimensions

    def intersection_subset(self, other: '_QidShapeSet'):
        """Return a subset of the intersection with other qid shape set."""
        explicit_qid_shapes = self.explicit_qid_shapes & other.explicit_qid_shapes
        unfactorized_total_dimension = None
        min_qudit_dimensions = None

        if self.explicit_qid_shapes and other.unfactorized_total_dimension is not None:
            explicit_qid_shapes |= _intersection_explicit_with_unfactorized_qid_shapes(
                self.explicit_qid_shapes, other.unfactorized_total_dimension
            )

        if self.explicit_qid_shapes and other.min_qudit_dimensions:
            explicit_qid_shapes |= _intersection_explicit_with_min_qudit_dims_qid_shapes(
                self.explicit_qid_shapes, other.min_qudit_dimensions
            )

        if self.unfactorized_total_dimension is not None and other.explicit_qid_shapes:
            explicit_qid_shapes |= _intersection_explicit_with_unfactorized_qid_shapes(
                other.explicit_qid_shapes, self.unfactorized_total_dimension
            )

        if self.unfactorized_total_dimension == other.unfactorized_total_dimension:
            unfactorized_total_dimension = self.unfactorized_total_dimension

        if self.min_qudit_dimensions is not None and other.explicit_qid_shapes:
            explicit_qid_shapes |= _intersection_explicit_with_min_qudit_dims_qid_shapes(
                other.explicit_qid_shapes, self.min_qudit_dimensions
            )

        if self.min_qudit_dimensions is not None and other.min_qudit_dimensions is not None:
            min_qudit_dimensions = _intersection_min_qudit_dims_qid_shapes(
                self.min_qudit_dimensions, other.min_qudit_dimensions
            )

        # Above, we only intersected 7 pairs of sets even though there are 9 possible pairs.
        # We omitted the 2 pairs involving unfactorized total dimension and min qudit dimensions
        # because that calculation is inefficient.

        return _QidShapeSet(
            explicit_qid_shapes=explicit_qid_shapes,
            unfactorized_total_dimension=unfactorized_total_dimension,
            min_qudit_dimensions=min_qudit_dimensions,
        )

    def _raise_value_error_if_ambiguous(self) -> None:
        """Raise an error if the qid shape is ambiguous and cannot be inferred."""
        if self.min_qudit_dimensions is not None:
            raise ValueError(
                'Qid shape is ambiguous: '
                f'Could be any shape on {len(self.min_qudit_dimensions)} qudits '
                'with the corresponding qudit dimensions being at least '
                f'{self.min_qudit_dimensions}.'
            )
        if len(self.explicit_qid_shapes) > 1:
            # coverage: ignore
            raise ValueError(
                f'Qid shape is ambiguous: Could be any one of {self.explicit_qid_shapes}.'
            )
        if self.explicit_qid_shapes and self.unfactorized_total_dimension is not None:
            explicit_shape = next(iter(self.explicit_qid_shapes))
            raise ValueError(
                'Qid shape is ambiguous: '
                f'Could be {explicit_shape} or any shape compatible with '
                f'a Hilbert space dimension of {self.unfactorized_total_dimension}.'
            )

    def infer_qid_shape(self) -> Optional[Tuple[int, ...]]:
        """Return a qid shape from this set, or None."""
        self._raise_value_error_if_ambiguous()
        if self.unfactorized_total_dimension is not None:
            return _infer_qid_shape_from_dimension(self.unfactorized_total_dimension)
        if len(self.explicit_qid_shapes) == 0:
            return None
        return self.explicit_qid_shapes.pop()


def _intersection_explicit_with_unfactorized_qid_shapes(
    explicit_qid_shapes: Set[Tuple[int, ...]], unfactorized_total_dimension: int
) -> Set[Tuple[int, ...]]:
    return {
        qid_shape
        for qid_shape in explicit_qid_shapes
        if np.prod(qid_shape, dtype=np.int64) == unfactorized_total_dimension
    }


def _intersection_explicit_with_min_qudit_dims_qid_shapes(
    explicit_qid_shapes: Set[Tuple[int, ...]], min_qudit_dimensions: Tuple[int, ...]
) -> Set[Tuple[int, ...]]:
    return {
        qid_shape
        for qid_shape in explicit_qid_shapes
        if len(qid_shape) == len(min_qudit_dimensions)
        and all(dim1 >= dim2 for dim1, dim2 in zip(qid_shape, min_qudit_dimensions))
    }


def _intersection_min_qudit_dims_qid_shapes(
    min_qudit_dimensions1: Tuple[int, ...], min_qudit_dimensions2: Tuple[int, ...]
) -> Optional[Tuple[int, ...]]:
    if len(min_qudit_dimensions1) == len(min_qudit_dimensions2):
        return tuple(
            max(dim1, dim2) for dim1, dim2 in zip(min_qudit_dimensions1, min_qudit_dimensions2)
        )
    return None


def bloch_vector_from_state_vector(
    state_vector: np.ndarray, index: int, qid_shape: Optional[Tuple[int, ...]] = None
) -> np.ndarray:
    """Returns the bloch vector of a qubit.

    Calculates the bloch vector of the qubit at index in the state vector,
    assuming state vector follows the standard Kronecker convention of
    numpy.kron.

    Args:
        state_vector: A sequence representing a state vector in which
            the ordering mapping to qubits follows the standard Kronecker
            convention of numpy.kron (big-endian).
        index: index of qubit who's bloch vector we want to find.
            follows the standard Kronecker convention of numpy.kron.
        qid_shape: specifies the dimensions of the qudits for the input
            `state_vector`.  If not specified, qubits are assumed and the
            `state_vector` must have a dimension a power of two.
            The qudit at `index` must be a qubit.

    Returns:
        A length 3 numpy array representing the qubit's bloch vector.

    Raises:
        ValueError: if the size of `state_vector `is not a power of 2 and the
            shape is not given or if the shape is given and `state_vector` has
            a size that contradicts this shape.
        IndexError: if index is out of range for the number of qubits or qudits
            corresponding to `state_vector`.
    """
    rho = density_matrix_from_state_vector(state_vector, [index], qid_shape=qid_shape)
    v = np.zeros(3, dtype=np.float32)
    v[0] = 2 * np.real(rho[0][1])
    v[1] = 2 * np.imag(rho[1][0])
    v[2] = np.real(rho[0][0] - rho[1][1])

    return v


def density_matrix_from_state_vector(
    state_vector: np.ndarray,
    indices: Optional[Iterable[int]] = None,
    qid_shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    r"""Returns the density matrix of the state vector.

    Calculate the density matrix for the system on the given qubit indices,
    with the qubits not in indices that are present in state vector traced out.
    If indices is None the full density matrix for `state_vector` is returned.
    We assume `state_vector` follows the standard Kronecker convention of
    numpy.kron (big-endian).

    For example:
    state_vector = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex64)
    indices = None
    gives us

        $$
        \rho = \begin{bmatrix}
                0.5 & 0.5 \\
                0.5 & 0.5
        \end{bmatrix}
        $$

    Args:
        state_vector: A sequence representing a state vector in which
            the ordering mapping to qubits follows the standard Kronecker
            convention of numpy.kron (big-endian).
        indices: list containing indices for qubits that you would like
            to include in the density matrix (i.e.) qubits that WON'T
            be traced out. follows the standard Kronecker convention of
            numpy.kron.
        qid_shape: specifies the dimensions of the qudits for the input
            `state_vector`.  If not specified, qubits are assumed and the
            `state_vector` must have a dimension a power of two.

    Returns:
        A numpy array representing the density matrix.

    Raises:
        ValueError: if the size of `state_vector` is not a power of 2 and the
            shape is not given or if the shape is given and `state_vector`
            has a size that contradicts this shape.
        IndexError: if the indices are out of range for the number of qubits
            corresponding to `state_vector`.
    """
    shape = validate_qid_shape(state_vector, qid_shape)
    n_qubits = len(shape)

    if indices is None:
        return np.outer(state_vector, np.conj(state_vector))

    indices = list(indices)
    validate_indices(n_qubits, indices)

    state_vector = np.asarray(state_vector).reshape(shape)

    sum_inds = np.array(range(n_qubits))
    sum_inds[indices] += n_qubits

    rho = np.einsum(
        state_vector,
        list(range(n_qubits)),
        np.conj(state_vector),
        sum_inds.tolist(),
        indices + sum_inds[indices].tolist(),
    )
    new_shape = np.prod([shape[i] for i in indices], dtype=np.int64)

    return rho.reshape((new_shape, new_shape))


def dirac_notation(
    state_vector: np.ndarray, decimals: int = 2, qid_shape: Optional[Tuple[int, ...]] = None
) -> str:
    """Returns the state vector as a string in Dirac notation.

    For example:

        state_vector = np.array([1/np.sqrt(2), 1/np.sqrt(2)],
                                dtype=np.complex64)
        print(dirac_notation(state_vector)) -> 0.71|0⟩ + 0.71|1⟩

    Args:
        state_vector: A sequence representing a state vector in which
            the ordering mapping to qubits follows the standard Kronecker
            convention of numpy.kron (big-endian).
        decimals: How many decimals to include in the pretty print.
        qid_shape: specifies the dimensions of the qudits for the input
            `state_vector`.  If not specified, qubits are assumed and the
            `state_vector` must have a dimension a power of two.

    Returns:
        A pretty string consisting of a sum of computational basis kets
        and non-zero floats of the specified accuracy.
    """
    if qid_shape is None:
        qid_shape = (2,) * (len(state_vector).bit_length() - 1)

    digit_separator = '' if max(qid_shape, default=0) < 10 else ','
    perm_list = [
        digit_separator.join(seq)
        for seq in itertools.product(*((str(i) for i in range(d)) for d in qid_shape))
    ]
    components = []
    ket = "|{}⟩"
    for x in range(len(perm_list)):
        format_str = "({:." + str(decimals) + "g})"
        val = round(state_vector[x].real, decimals) + 1j * round(state_vector[x].imag, decimals)

        if round(val.real, decimals) == 0 and round(val.imag, decimals) != 0:
            val = val.imag
            format_str = "{:." + str(decimals) + "g}j"
        elif round(val.imag, decimals) == 0 and round(val.real, decimals) != 0:
            val = val.real
            format_str = "{:." + str(decimals) + "g}"
        if val != 0:
            if (
                round(state_vector[x].real, decimals) == 1
                and round(state_vector[x].imag, decimals) == 0
            ):
                components.append(ket.format(perm_list[x]))
            else:
                components.append((format_str + ket).format(val, perm_list[x]))
    if not components:
        return '0'

    return ' + '.join(components).replace(' + -', ' - ')


def to_valid_state_vector(
    state_rep: 'cirq.STATE_VECTOR_LIKE',
    num_qubits: Optional[int] = None,
    *,  # Force keyword arguments
    qid_shape: Optional[Sequence[int]] = None,
    dtype: Optional['DTypeLike'] = None,
    atol: float = 1e-7,
) -> np.ndarray:
    """Verifies the state_rep is valid and converts it to ndarray form.

    This method is used to support passing in an integer representing a
    computational basis state or a full state vector as a representation of
    a pure state.

    Args:
        state_rep: If an int, the state vector returned is the state vector
            corresponding to a computational basis state. If a numpy array
            this is the full state vector. Both of these are validated for
            the given number of qubits, and the state must be properly
            normalized and of the appropriate dtype.
        num_qubits: The number of qubits for the state vector. The state_rep
            must be valid for this number of qubits.
        qid_shape: The expected qid shape of the state vector. Specify this
            argument when using qudits.
        dtype: The numpy dtype of the state vector, will be used when creating
            the state for a computational basis state, or validated against if
            state_rep is a numpy array.
        atol: Numerical tolerance for verifying that the norm of the state
            vector is close to 1.

    Returns:
        A numpy ndarray corresponding to the state vector on the given number of
        qubits.

    Raises:
        ValueError: if `state_vector` is not valid or
            num_qubits != len(qid_shape).
    """
    if isinstance(state_rep, value.ProductState):
        num_qubits = len(state_rep)

    # Check shape.
    if num_qubits is None and qid_shape is None:
        try:
            qid_shape = infer_qid_shape(state_rep)
        except:
            raise ValueError(
                'Failed to infer the qid shape of the given state. '
                'Please specify the qid shape explicitly using either the '
                '`num_qubits` or `qid_shape` argument.'
            )
    if qid_shape is None:
        qid_shape = (2,) * cast(int, num_qubits)
    else:
        qid_shape = tuple(qid_shape)
    if num_qubits is None:
        num_qubits = len(qid_shape)
    if num_qubits != len(qid_shape):
        raise ValueError(
            'num_qubits != len(qid_shape). num_qubits is <{!r}>. '
            'qid_shape is <{!r}>.'.format(num_qubits, qid_shape)
        )

    if isinstance(state_rep, np.ndarray):
        state_rep = np.copy(state_rep)
    state = quantum_state(state_rep, qid_shape, validate=True, dtype=dtype, atol=atol)
    return cast(np.ndarray, state.state_vector())


def _qudit_values_to_state_tensor(
    *, state_vector: np.ndarray, qid_shape: Tuple[int, ...], dtype: Optional['DTypeLike']
) -> np.ndarray:

    for i in range(len(qid_shape)):
        s = state_vector[i]
        q = qid_shape[i]
        if not 0 <= s < q:
            raise ValueError(
                f'Qudit value {s} at index {i} is out of bounds for '
                f'qudit dimension {q}.\n'
                f'\n'
                f'qid_shape={qid_shape!r}\n'
                f'state={state_vector!r}\n'
            )

    if state_vector.dtype.kind[0] not in '?bBiu':
        raise ValueError(
            f'Expected a bool or int entry for each qudit in '
            f'`state`, because len(state) == len(qid_shape), '
            f'but got dtype {state_vector.dtype}.'
            f'\n'
            f'qid_shape={qid_shape!r}\n'
            f'state={state_vector!r}\n'
        )

    if dtype is None:
        dtype = DEFAULT_COMPLEX_DTYPE
    return one_hot(index=tuple(int(e) for e in state_vector), shape=qid_shape, dtype=dtype)


def validate_normalized_state_vector(
    state_vector: np.ndarray,
    *,  # Force keyword arguments
    qid_shape: Tuple[int, ...],
    dtype: Optional['DTypeLike'] = None,
    atol: float = 1e-7,
) -> None:
    """Checks that the given state vector is valid.

    Args:
        state_vector: The state vector to validate.
        qid_shape: The expected qid shape of the state.
        dtype: The expected dtype of the state.
        atol: Absolute numerical tolerance.

    Raises:
        ValueError: State has invalid dtype.
        ValueError: State has incorrect size.
        ValueError: State is not normalized.
    """
    if dtype and state_vector.dtype != dtype:
        raise ValueError(
            'state_vector has invalid dtype. Expected {} but was {}'.format(
                dtype, state_vector.dtype
            )
        )
    if state_vector.size != np.prod(qid_shape, dtype=np.int64):
        raise ValueError(
            'state_vector has incorrect size. Expected {} but was {}.'.format(
                np.prod(qid_shape, dtype=np.int64), state_vector.size
            )
        )
    norm = np.sum(np.abs(state_vector) ** 2)
    if not np.isclose(norm, 1, atol=atol):
        raise ValueError(f'State_vector is not normalized instead had norm {norm}')


def validate_qid_shape(
    state_vector: np.ndarray, qid_shape: Optional[Tuple[int, ...]]
) -> Tuple[int, ...]:
    """Validates the size of the given `state_vector` against the given shape.

    Returns:
        The qid shape.

    Raises:
        ValueError: if the size of `state_vector` does not match that given in
            `qid_shape` or if `qid_state` is not given if `state_vector` does
            not have a dimension that is a power of two.
    """
    size = state_vector.size
    if qid_shape is None:
        qid_shape = (2,) * (size.bit_length() - 1)
    if size != np.prod(qid_shape, dtype=np.int64):
        raise ValueError(
            'state_vector.size ({}) is not a power of two or is not a product '
            'of the qid shape {!r}.'.format(size, qid_shape)
        )
    return qid_shape


def validate_indices(num_qubits: int, indices: Sequence[int]) -> None:
    """Validates that the indices have values within range of num_qubits."""
    if any(index < 0 for index in indices):
        raise IndexError(f'Negative index in indices: {indices}')
    if any(index >= num_qubits for index in indices):
        raise IndexError(
            f'Out of range indices, must be less than number of qubits but was {indices}'
        )


def to_valid_density_matrix(
    density_matrix_rep: Union[np.ndarray, 'cirq.STATE_VECTOR_LIKE'],
    num_qubits: Optional[int] = None,
    *,  # Force keyword arguments
    qid_shape: Optional[Tuple[int, ...]] = None,
    dtype: Optional['DTypeLike'] = None,
    atol: float = 1e-7,
) -> np.ndarray:
    """Verifies the density_matrix_rep is valid and converts it to ndarray form.

    This method is used to support passing a matrix, a state vector,
    or a computational basis state as a representation of a state.

    Args:
        density_matrix_rep: If a numpy array, if it is of rank 2 (a matrix),
            then this is the density matrix. If it is a numpy array of rank 1
            (a vector) then this is a state vector. If this is an int,
            then this is the computation basis state.
        num_qubits: The number of qubits for the density matrix. The
            density_matrix_rep must be valid for this number of qubits.
        qid_shape: The qid shape of the state vector. Specify this argument
            when using qudits.
        dtype: The numpy dtype of the density matrix, will be used when creating
            the state for a computational basis state (int), or validated
            against if density_matrix_rep is a numpy array.
        atol: Numerical tolerance for verifying density matrix properties.

    Returns:
        A numpy matrix corresponding to the density matrix on the given number
        of qubits. Note that this matrix may share memory with the input
        `density_matrix_rep`.

    Raises:
        ValueError if the density_matrix_rep is not valid.
    """
    qid_shape = _qid_shape_from_args(num_qubits, qid_shape)
    if isinstance(density_matrix_rep, np.ndarray) and density_matrix_rep.ndim == 2:
        validate_density_matrix(density_matrix_rep, qid_shape=qid_shape, dtype=dtype, atol=atol)
        return density_matrix_rep

    state_vector = to_valid_state_vector(
        density_matrix_rep, len(qid_shape), qid_shape=qid_shape, dtype=dtype
    )
    return np.outer(state_vector, np.conj(state_vector))


def validate_density_matrix(
    density_matrix: np.ndarray,
    *,  # Force keyword arguments
    qid_shape: Tuple[int, ...],
    dtype: Optional['DTypeLike'] = None,
    atol: float = 1e-7,
) -> None:
    """Checks that the given density matrix is valid.

    Args:
        density_matrix: The density matrix to validate.
        qid_shape: The expected qid shape.
        dtype: The expected dtype.
        atol: Absolute numerical tolerance.

    Raises:
        ValueError: The density matrix does not have the correct dtype.
        ValueError: The density matrix does not have the correct shape.
            It should be a square matrix with dimension prod(qid_shape).
        ValueError: The density matrix is not Hermitian.
        ValueError: The density matrix does not have trace 1.
        ValueError: The density matrix is not positive semidefinite.
    """
    if dtype and density_matrix.dtype != dtype:
        raise ValueError(
            f'Incorrect dtype for density matrix: Expected {dtype} '
            f'but has dtype {density_matrix.dtype}.'
        )
    expected_shape = (np.prod(qid_shape, dtype=np.int64),) * 2
    if density_matrix.shape != expected_shape:
        raise ValueError(
            f'Incorrect shape for density matrix: Expected {expected_shape} '
            f'but has shape {density_matrix.shape}.'
        )
    if not np.allclose(density_matrix, density_matrix.conj().T, atol=atol):
        raise ValueError('The density matrix is not hermitian.')
    trace = np.trace(density_matrix)
    if not np.isclose(trace, 1.0, atol=atol):
        raise ValueError(f'Density matrix does not have trace 1. Instead, it has trace {trace}.')
    if not np.all(np.linalg.eigvalsh(density_matrix) > -atol):
        raise ValueError('The density matrix is not positive semidefinite.')


def _qid_shape_from_args(
    num_qubits: Optional[int], qid_shape: Optional[Tuple[int, ...]]
) -> Tuple[int, ...]:
    """Returns either `(2,) * num_qubits` or `qid_shape`.

    Raises:
        ValueError: If both arguments are None or their values disagree.
    """
    if num_qubits is None and qid_shape is None:
        raise ValueError(
            'Either the num_qubits or qid_shape argument must be specified. Both were None.'
        )
    if num_qubits is None:
        return cast(Tuple[int, ...], qid_shape)
    if qid_shape is None:
        return (2,) * num_qubits
    if len(qid_shape) != num_qubits:
        raise ValueError(
            'num_qubits != len(qid_shape). num_qubits was {!r}. '
            'qid_shape was {!r}.'.format(num_qubits, qid_shape)
        )
    return qid_shape


def one_hot(
    *,
    index: Union[None, int, Sequence[int]] = None,
    shape: Union[int, Sequence[int]],
    value: Any = 1,
    dtype: 'DTypeLike',
) -> np.ndarray:
    """Returns a numpy array with all 0s and a single non-zero entry(default 1).

    Args:
        index: The index that should store the `value` argument instead of 0.
            If not specified, defaults to the start of the array.
        shape: The shape of the array.
        value: The hot value to place at `index` in the result.
        dtype: The dtype of the array.

    Returns:
        The created numpy array.
    """
    if index is None:
        index = 0 if isinstance(shape, int) else (0,) * len(shape)
    result = np.zeros(shape=shape, dtype=dtype)
    result[index] = value
    return result


def eye_tensor(half_shape: Tuple[int, ...], *, dtype: 'DTypeLike') -> np.ndarray:
    """Returns an identity matrix reshaped into a tensor.

    Args:
        half_shape: A tuple representing the number of quantum levels of each
            qubit the returned matrix applies to.  `half_shape` is (2, 2, 2) for
            a three-qubit identity operation tensor.
        dtype: The numpy dtype of the new array.

    Returns:
        The created numpy array with shape `half_shape + half_shape`.
    """
    identity = np.eye(np.prod(half_shape, dtype=np.int64).item(), dtype=dtype)
    identity.shape = half_shape * 2
    return identity
