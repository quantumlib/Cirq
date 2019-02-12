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

from typing import List, Tuple, Type, Union

import numpy as np

from cirq import linalg
from cirq.sim import wave_function


def to_valid_density_matrix(
    density_matrix_rep: Union[int, np.ndarray],
    num_qubits: int,
    dtype: Type[np.number] = np.complex64) -> np.ndarray:
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
        dtype: The numpy dtype of the density matrix, will be used when creating
            the state for a computational basis state (int), or validated
            against if density_matrix_rep is a numpy array.

    Returns:
        A numpy matrix corresponding to the density matrix on the given number
        of qubits.

    Raises:
        ValueError if the density_matrix_rep is not valid.
    """
    if (isinstance(density_matrix_rep, np.ndarray)
        and density_matrix_rep.ndim == 2):
        if density_matrix_rep.shape != (2 ** num_qubits, 2 ** num_qubits):
            raise ValueError(
                'Density matrix was not square and of size 2 ** num_qubit, '
                'instead was {}'.format(density_matrix_rep.shape))
        if not np.allclose(density_matrix_rep,
                           np.transpose(np.conj(density_matrix_rep))):
            raise ValueError('The density matrix is not hermitian.')
        if not np.isclose(np.trace(density_matrix_rep), 1.0):
            raise ValueError(
                'Density matrix did not have trace 1 but instead {}'.format(
                    np.trace(density_matrix_rep)))
        if density_matrix_rep.dtype != dtype:
            raise ValueError(
                'Density matrix had dtype {} but expected {}'.format(
                    density_matrix_rep.dtype, dtype))
        if not np.all(np.linalg.eigvalsh(density_matrix_rep) > -1e-8):
            raise ValueError('The density matrix is not positive semidefinite.')
        return density_matrix_rep

    state_vector = wave_function.to_valid_state_vector(density_matrix_rep,
                                                       num_qubits, dtype)
    return np.outer(state_vector, np.conj(state_vector))


def sample_density_matrix(
    density_matrix: np.ndarray,
    indices: List[int],
    repetitions: int=1) -> np.ndarray:
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
        repetitions: The number of times to sample the density matrix.

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
    num_qubits = _validate_num_qubits(density_matrix)
    _validate_indices(num_qubits, indices)

    if repetitions == 0 or len(indices) == 0:
        return np.zeros(shape=(repetitions, len(indices)))

    # Calculate the measurement probabilities.
    probs = _probs(density_matrix, indices, num_qubits)

    # We now have the probability vector, correctly ordered, so sample over
    # it. Note that we us ints here, since numpy's choice does not allow for
    # choosing from a list of tuples or list of lists.
    result = np.random.choice(len(probs), size=repetitions, p=probs)
    # Convert to bools and rearrange to match repetition being the outer list.
    return np.transpose([(1 & (result >> i)).astype(np.bool) for i in
                         range(len(indices))])


def measure_density_matrix(
    density_matrix: np.ndarray,
    indices: List[int],
    out: np.ndarray = None) -> Tuple[List[bool], np.ndarray]:
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
        out: An optional place to store the result. If `out` is the same as
            the `density_matrix` parameter, then `density_matrix` will be
            modified inline. If `out` is not None, then the result is put into
            `out`.  If `out` is None a new value will be allocated. In all of
            these cases `out` will be the same as the returned ndarray of the
            method. The shape and dtype of `out` will match that of
            `density_matrix` if `out` is None, otherwise it will match the
            shape and dtype of `out`.

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
    num_qubits = _validate_num_qubits(density_matrix)
    _validate_indices(num_qubits, indices)

    if len(indices) == 0:
        if out is None:
            out = np.copy(density_matrix)
        elif out is not density_matrix:
            np.copyto(dst=out, src=density_matrix)
        return ([], out)
        # Final else: if out is matrix then matrix will be modified in place.

    # Cache initial shape.
    initial_shape = density_matrix.shape

    # Calculate the measurement probabilities and then make the measurement.
    probs = _probs(density_matrix, indices, num_qubits)
    result = np.random.choice(len(probs), p=probs)
    measurement_bits = [(1 & (result >> i)) for i in range(len(indices))]

    # Calculate the slice for the measurement result.
    result_slice = linalg.slice_for_qubits_equal_to(indices, result,
                                                    num_qubits=num_qubits)
    # Create a mask which is False for only the slice.
    mask = np.ones([2] * 2 * num_qubits, dtype=bool)
    # Remove ellipses from last element of
    mask[result_slice * 2] = False

    if out is None:
        out = np.copy(density_matrix)
    elif out is not density_matrix:
        np.copyto(dst=out, src=density_matrix)
    # Final else: if out is matrix then matrix will be modified in place.

    # Potentially reshape to tensor, and then set masked values to 0.
    out.shape = [2] * num_qubits * 2
    out[mask] = 0

    # Restore original shape (if necessary) and renormalize.
    out.shape = initial_shape
    out /= probs[result]

    return measurement_bits, out


def _probs(density_matrix: np.ndarray, indices: List[int],
    num_qubits: int) -> List[float]:
    """Returns the probabilities for a measurement on the given indices."""
    # Only diagonal elements matter.
    all_probs = np.diagonal(
        np.reshape(density_matrix, (2 ** num_qubits, 2 ** num_qubits)))
    # Shape into a tensor
    tensor = np.reshape(all_probs, [2] * num_qubits)

    # Calculate the probabilities for measuring the particular results.
    probs = [
        np.sum(np.abs(tensor[linalg.slice_for_qubits_equal_to(indices, b)]))
        for b in range(2 ** len(indices))]

    # To deal with rounding issues, ensure that the probabilities sum to 1.
    probs /= np.sum(probs) # type: ignore
    return probs


def _validate_num_qubits(density_matrix: np.ndarray) -> int:
    """Validates that matrix's shape is a valid shape for qubits.
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


def _validate_indices(num_qubits: int, indices: List[int]) -> None:
    """Validates that the indices have values within range of num_qubits."""
    if any(index < 0 for index in indices):
        raise IndexError('Negative index in indices: {}'.format(indices))
    if any(index >= num_qubits for index in indices):
        raise IndexError('Out of range indices, must be less than number of '
                         'qubits but was {}'.format(indices))
