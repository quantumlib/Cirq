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
"""Code to handle density matrices."""

from typing import cast, List, Optional, Tuple, Type, Union

import numpy as np
from scipy.stats import entropy

from cirq import linalg, value
from cirq.sim import wave_function


def to_valid_density_matrix(
        density_matrix_rep: Union[int, np.ndarray],
        num_qubits: Optional[int] = None,
        *,  # Force keyword arguments
        qid_shape: Optional[Tuple[int, ...]] = None,
        dtype: Type[np.number] = np.complex64,
        atol: float = 1e-7) -> np.ndarray:
    """Verifies the density_matrix_rep is valid and converts it to ndarray form.

    This method is used to support passing a matrix, a vector (wave function),
    or a computational basis state as a representation of a state.

    Args:
        density_matrix_rep: If an numpy array, if it is of rank 2 (a matrix),
            then this is the density matrix. If it is a numpy array of rank 1
            (a vector) then this is a wave function. If this is an int,
            then this is the computation basis state.
        num_qubits: The number of qubits for the density matrix. The
            density_matrix_rep must be valid for this number of qubits.
        qid_shape: The qid shape of the state vector.  Specify this argument
            when using qudits.
        dtype: The numpy dtype of the density matrix, will be used when creating
            the state for a computational basis state (int), or validated
            against if density_matrix_rep is a numpy array.
        atol: Numerical tolerance for verifying density matrix properties.

    Returns:
        A numpy matrix corresponding to the density matrix on the given number
        of qubits.

    Raises:
        ValueError if the density_matrix_rep is not valid.
    """
    qid_shape = _qid_shape_from_args(num_qubits, qid_shape)
    if (isinstance(density_matrix_rep, np.ndarray)
        and density_matrix_rep.ndim == 2):
        if density_matrix_rep.shape != (np.prod(qid_shape, dtype=int),) * 2:
            raise ValueError(
                'Density matrix was not square and of size 2 ** num_qubit, '
                'instead was {}'.format(density_matrix_rep.shape))
        if not np.allclose(density_matrix_rep,
                           np.transpose(np.conj(density_matrix_rep)),
                           atol=atol):
            raise ValueError('The density matrix is not hermitian.')
        if not np.isclose(np.trace(density_matrix_rep), 1.0, atol=atol):
            raise ValueError(
                'Density matrix did not have trace 1 but instead {}'.format(
                    np.trace(density_matrix_rep)))
        if density_matrix_rep.dtype != dtype:
            raise ValueError(
                'Density matrix had dtype {} but expected {}'.format(
                    density_matrix_rep.dtype, dtype))
        if not np.all(np.linalg.eigvalsh(density_matrix_rep) > -atol):
            raise ValueError('The density matrix is not positive semidefinite.')
        return density_matrix_rep

    state_vector = wave_function.to_valid_state_vector(density_matrix_rep,
                                                       len(qid_shape),
                                                       qid_shape=qid_shape,
                                                       dtype=dtype)
    return np.outer(state_vector, np.conj(state_vector))


def sample_density_matrix(
        density_matrix: np.ndarray,
        indices: List[int],
        *,  # Force keyword arguments
        qid_shape: Optional[Tuple[int, ...]] = None,
        repetitions: int = 1,
        seed: value.RANDOM_STATE_LIKE = None) -> np.ndarray:
    """Samples repeatedly from measurements in the computational basis.

    Note that this does not modify the density_matrix.

    Args:
        density_matrix: The density matrix to be measured. This matrix is
            assumed to be positive semidefinite and trace one. The matrix is
            assumed to be of shape (2 ** integer, 2 ** integer) or
            (2, 2, ..., 2).
        indices: Which qubits are measured. The density matrix rows and columns
            are assumed to be supplied in big endian order. That is the
            xth index of v, when expressed as a bitstring, has its largest
            values in the 0th index.
        qid_shape: The qid shape of the density matrix.  Specify this argument
            when using qudits.
        repetitions: The number of times to sample the density matrix.
        seed: A seed for the pseudorandom number generator.

    Returns:
        Measurement results with True corresponding to the ``|1⟩`` state.
        The outer list is for repetitions, and the inner corresponds to
        measurements ordered by the supplied qubits. These lists
        are wrapped as an numpy ndarray.

    Raises:
        ValueError: ``repetitions`` is less than one or size of ``matrix`` is
            not a power of 2.
        IndexError: An index from ``indices`` is out of range, given the number
            of qubits corresponding to the density matrix.
    """
    if repetitions < 0:
        raise ValueError('Number of repetitions cannot be negative. Was {}'
                         .format(repetitions))
    if qid_shape is None:
        num_qubits = _validate_num_qubits(density_matrix)
        qid_shape = (2,) * num_qubits
    else:
        _validate_density_matrix_qid_shape(density_matrix, qid_shape)
        num_qubits = len(qid_shape)
    meas_shape = _indices_shape(qid_shape, indices)

    if repetitions == 0 or len(indices) == 0:
        return np.zeros(shape=(repetitions, len(indices)), dtype=np.int8)

    prng = value.parse_random_state(seed)

    # Calculate the measurement probabilities.
    probs = _probs(density_matrix, indices, qid_shape)

    # We now have the probability vector, correctly ordered, so sample over
    # it. Note that we us ints here, since numpy's choice does not allow for
    # choosing from a list of tuples or list of lists.
    result = prng.choice(len(probs), size=repetitions, p=probs)
    # Convert to individual qudit measurements.
    return np.array([
        value.big_endian_int_to_digits(result[i], base=meas_shape)
        for i in range(len(result))
    ],
                    dtype=np.int8)


def measure_density_matrix(density_matrix: np.ndarray,
                           indices: List[int],
                           qid_shape: Optional[Tuple[int, ...]] = None,
                           out: np.ndarray = None,
                           seed: value.RANDOM_STATE_LIKE = None
                          ) -> Tuple[List[int], np.ndarray]:
    """Performs a measurement of the density matrix in the computational basis.

    This does not modify `density_matrix` unless the optional `out` is
    `density_matrix`.

    Args:
        density_matrix: The density matrix to be measured. This matrix is
            assumed to be positive semidefinite and trace one. The matrix is
            assumed to be of shape (2 ** integer, 2 ** integer) or
            (2, 2, ..., 2).
        indices: Which qubits are measured. The matrix is assumed to be supplied
            in big endian order. That is the xth index of v, when expressed as
            a bitstring, has the largest values in the 0th index.
        qid_shape: The qid shape of the density matrix.  Specify this argument
            when using qudits.
        out: An optional place to store the result. If `out` is the same as
            the `density_matrix` parameter, then `density_matrix` will be
            modified inline. If `out` is not None, then the result is put into
            `out`.  If `out` is None a new value will be allocated. In all of
            these cases `out` will be the same as the returned ndarray of the
            method. The shape and dtype of `out` will match that of
            `density_matrix` if `out` is None, otherwise it will match the
            shape and dtype of `out`.
        seed: A seed for the pseudorandom number generator.

    Returns:
        A tuple of a list and an numpy array. The list is an array of booleans
        corresponding to the measurement values (ordered by the indices). The
        numpy array is the post measurement matrix. This matrix has the same
        shape and dtype as the input matrix.

    Raises:
        ValueError if the dimension of the matrix is not compatible with a
            matrix of n qubits.
        IndexError if the indices are out of range for the number of qubits
            corresponding to the density matrix.
    """
    if qid_shape is None:
        num_qubits = _validate_num_qubits(density_matrix)
        qid_shape = (2,) * num_qubits
    else:
        _validate_density_matrix_qid_shape(density_matrix, qid_shape)
        num_qubits = len(qid_shape)
    meas_shape = _indices_shape(qid_shape, indices)

    if len(indices) == 0:
        if out is None:
            out = np.copy(density_matrix)
        elif out is not density_matrix:
            np.copyto(dst=out, src=density_matrix)
        return ([], out)
        # Final else: if out is matrix then matrix will be modified in place.

    prng = value.parse_random_state(seed)

    # Cache initial shape.
    initial_shape = density_matrix.shape

    # Calculate the measurement probabilities and then make the measurement.
    probs = _probs(density_matrix, indices, qid_shape)
    result = prng.choice(len(probs), p=probs)
    measurement_bits = value.big_endian_int_to_digits(result, base=meas_shape)

    # Calculate the slice for the measurement result.
    result_slice = linalg.slice_for_qubits_equal_to(
        indices, big_endian_qureg_value=result, qid_shape=qid_shape)
    # Create a mask which is False for only the slice.
    mask = np.ones(qid_shape * 2, dtype=bool)
    # Remove ellipses from last element of
    mask[result_slice * 2] = False

    if out is None:
        out = np.copy(density_matrix)
    elif out is not density_matrix:
        np.copyto(dst=out, src=density_matrix)
    # Final else: if out is matrix then matrix will be modified in place.

    # Potentially reshape to tensor, and then set masked values to 0.
    out.shape = qid_shape * 2
    out[mask] = 0

    # Restore original shape (if necessary) and renormalize.
    out.shape = initial_shape
    out /= probs[result]

    return measurement_bits, out


def _probs(density_matrix: np.ndarray, indices: List[int],
           qid_shape: Tuple[int, ...]) -> np.ndarray:
    """Returns the probabilities for a measurement on the given indices."""
    # Only diagonal elements matter.
    all_probs = np.diagonal(
        np.reshape(density_matrix, (np.prod(qid_shape, dtype=int),) * 2))
    # Shape into a tensor
    tensor = np.reshape(all_probs, qid_shape)

    # Calculate the probabilities for measuring the particular results.
    if len(indices) == len(qid_shape):
        # We're measuring every qudit, so no need for fancy indexing
        probs = np.abs(tensor)
        probs = np.transpose(probs, indices)
        probs = np.reshape(probs, np.prod(probs.shape))
    else:
        # Fancy indexing required
        meas_shape = tuple(qid_shape[i] for i in indices)
        probs = np.abs([
            tensor[linalg.slice_for_qubits_equal_to(indices,
                                                    big_endian_qureg_value=b,
                                                    qid_shape=qid_shape)]
            for b in range(np.prod(meas_shape, dtype=int))
        ])
        probs = np.sum(probs, axis=tuple(range(1, len(probs.shape))))

    # To deal with rounding issues, ensure that the probabilities sum to 1.
    probs /= np.sum(probs)
    return probs


def _qid_shape_from_args(num_qubits: Optional[int],
                         qid_shape: Optional[Tuple[int, ...]]
                        ) -> Tuple[int, ...]:
    """Returns either `(2,) * num_qubits` or `qid_shape`.

    Raises:
        ValueError: If both arguments are None or their values disagree.
    """
    if num_qubits is None and qid_shape is None:
        raise ValueError('Either the num_qubits or qid_shape argument must be '
                         'specified. Both were None.')
    if num_qubits is None:
        return cast(Tuple[int, ...], qid_shape)
    if qid_shape is None:
        return (2,) * num_qubits
    if len(qid_shape) != num_qubits:
        raise ValueError('num_qubits != len(qid_shape). num_qubits was {!r}. '
                         'qid_shape was {!r}.'.format(num_qubits, qid_shape))
    return qid_shape


def _validate_density_matrix_qid_shape(density_matrix: np.array,
                                       qid_shape: Tuple[int, ...]
                                      ) -> Tuple[int, ...]:
    """Validates that a tensor's shape is a valid shape for qids and returns the
    qid shape.
    """
    shape = density_matrix.shape
    if len(shape) == 2:
        if np.prod(qid_shape, dtype=int)**2 != np.prod(shape, dtype=int):
            raise ValueError(
                'Matrix size does not match qid shape {!r}. Got matrix with '
                'shape {!r}. Expected {!r}.'.format(
                    qid_shape, shape, np.prod(qid_shape, dtype=int)))
        return qid_shape
    if len(shape) % 2 != 0:
        raise ValueError('Tensor was not square. Shape was {}'.format(shape))
    left_shape = shape[:len(shape) // 2]
    right_shape = shape[len(shape) // 2:]
    if left_shape != right_shape:
        raise ValueError(
            "Tensor's left and right shape are not equal. Shape was {}".format(
                shape))
    return left_shape


def _validate_num_qubits(density_matrix: np.ndarray) -> int:
    """Validates that matrix's shape is a valid shape for qubits.

    This method only works on a qubit-only density matrix.  Use
    `_validate_density_matrix_qid_shape` otherwise.
    """
    shape = density_matrix.shape
    half_index = len(shape) // 2
    row_size = np.prod(shape[:half_index]) if len(shape) != 0 else 0
    col_size = np.prod(shape[half_index:]) if len(shape) != 0 else 0
    if row_size != col_size:
        raise ValueError(
            'Matrix was not square. Shape was {}'.format(shape))
    if row_size & (row_size - 1):
        raise ValueError(
            'Matrix could not be shaped into a square matrix with dimensions '
            'not a power of two. Shape was {}'.format(shape)
        )
    if len(shape) > 2 and not np.allclose(shape, 2):
        raise ValueError(
            'Matrix is a tensor of rank greater than 2, but had dimensions '
            'that are not powers of two. Shape was {}'.format(shape)
        )
    return int(row_size).bit_length() - 1


def _indices_shape(qid_shape: Tuple[int, ...],
                   indices: List[int]) -> Tuple[int, ...]:
    """Validates that the indices have values within range of `len(qid_shape)`.
    """
    if any(index < 0 for index in indices):
        raise IndexError('Negative index in indices: {}'.format(indices))
    if any(index >= len(qid_shape) for index in indices):
        raise IndexError('Out of range indices, must be less than number of '
                         'qubits but was {}'.format(indices))
    return tuple(qid_shape[i] for i in indices)


def von_neumann_entropy(density_matrix: np.ndarray) -> float:
    """Calculates von Neumann entropy of density matrix in bits.
    Args:
        density_matrix: The density matrix.
    Returns:
        The calculated von Neumann entropy.
    """
    eigenvalues = np.linalg.eigvalsh(density_matrix)
    return entropy(abs(eigenvalues), base=2)
