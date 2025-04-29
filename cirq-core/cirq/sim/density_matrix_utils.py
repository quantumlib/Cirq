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

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from cirq import linalg, value
from cirq.sim import simulation_utils

if TYPE_CHECKING:
    import cirq


def sample_density_matrix(
    density_matrix: np.ndarray,
    indices: Sequence[int],
    *,  # Force keyword arguments
    qid_shape: Optional[Tuple[int, ...]] = None,
    repetitions: int = 1,
    seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
) -> np.ndarray:
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
        are wrapped as a numpy ndarray.

    Raises:
        ValueError: ``repetitions`` is less than one or size of ``matrix`` is
            not a power of 2.
        IndexError: An index from ``indices`` is out of range, given the number
            of qubits corresponding to the density matrix.
    """
    if repetitions < 0:
        raise ValueError(f'Number of repetitions cannot be negative. Was {repetitions}')
    if qid_shape is None:
        num_qubits = _validate_num_qubits(density_matrix)
        qid_shape = (2,) * num_qubits
    else:
        _validate_density_matrix_qid_shape(density_matrix, qid_shape)
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
    return np.array(
        [value.big_endian_int_to_digits(result[i], base=meas_shape) for i in range(len(result))],
        dtype=np.int8,
    )


def measure_density_matrix(
    density_matrix: np.ndarray,
    indices: Sequence[int],
    qid_shape: Optional[Tuple[int, ...]] = None,
    out: Optional[np.ndarray] = None,
    seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
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
        A tuple of a list and a numpy array. The list is an array of booleans
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
    meas_shape = _indices_shape(qid_shape, indices)

    arrout: np.ndarray
    if out is None:
        arrout = np.copy(density_matrix)
    elif out is density_matrix:
        arrout = density_matrix
    else:
        np.copyto(dst=out, src=density_matrix)
        arrout = out

    if len(indices) == 0:
        return ([], arrout)

    prng = value.parse_random_state(seed)

    # Cache initial shape.
    initial_shape = density_matrix.shape

    # Calculate the measurement probabilities and then make the measurement.
    probs = _probs(density_matrix, indices, qid_shape)
    result = prng.choice(len(probs), p=probs)
    measurement_bits = value.big_endian_int_to_digits(result, base=meas_shape)

    # Calculate the slice for the measurement result.
    result_slice = linalg.slice_for_qubits_equal_to(
        indices, big_endian_qureg_value=result, qid_shape=qid_shape
    )
    # Create a mask which is False for only the slice.
    mask = np.ones(qid_shape * 2, dtype=bool)
    # Remove ellipses from last element of
    mask[result_slice * 2] = False

    # Potentially reshape to tensor, and then set masked values to 0.
    arrout.shape = qid_shape * 2
    arrout[mask] = 0

    # Restore original shape (if necessary) and renormalize.
    arrout.shape = initial_shape
    arrout /= probs[result]

    return measurement_bits, arrout


def _probs(
    density_matrix: np.ndarray, indices: Sequence[int], qid_shape: Tuple[int, ...]
) -> np.ndarray:
    """Returns the probabilities for a measurement on the given indices."""
    # Only diagonal elements matter.
    all_probs = np.diagonal(np.reshape(density_matrix, (np.prod(qid_shape, dtype=np.int64),) * 2))

    return simulation_utils.state_probabilities_by_indices(all_probs.real, indices, qid_shape)


def _validate_density_matrix_qid_shape(
    density_matrix: np.ndarray, qid_shape: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Validates that a tensor's shape is a valid shape for qids and returns the
    qid shape.
    """
    shape = density_matrix.shape
    if len(shape) == 2:
        if np.prod(qid_shape, dtype=np.int64) ** 2 != np.prod(shape, dtype=np.int64):
            raise ValueError(
                f'Matrix size does not match qid shape {qid_shape!r}. Got matrix with '
                f'shape {shape!r}. Expected {np.prod(qid_shape, dtype=np.int64)!r}.'
            )
        return qid_shape
    if len(shape) % 2 != 0:
        raise ValueError(f'Tensor was not square. Shape was {shape}')
    left_shape = shape[: len(shape) // 2]
    right_shape = shape[len(shape) // 2 :]
    if left_shape != right_shape:
        raise ValueError(f"Tensor's left and right shape are not equal. Shape was {shape}")
    return left_shape


def _validate_num_qubits(density_matrix: np.ndarray) -> int:
    """Validates that matrix's shape is a valid shape for qubits.

    This method only works on a qubit-only density matrix.  Use
    `_validate_density_matrix_qid_shape` otherwise.
    """
    shape = density_matrix.shape
    half_index = len(shape) // 2
    row_size = np.prod(shape[:half_index]).item() if shape else 0
    col_size = np.prod(shape[half_index:]).item() if shape else 0
    if row_size != col_size:
        raise ValueError(f'Matrix was not square. Shape was {shape}')
    if row_size & (row_size - 1):
        raise ValueError(
            'Matrix could not be shaped into a square matrix with dimensions '
            f'that are a power of two. Shape was {shape}'
        )
    if len(shape) > 2 and not np.allclose(shape, 2):
        raise ValueError(
            'Matrix is a tensor of rank greater than 2, but had dimensions '
            f'that are not powers of two. Shape was {shape}'
        )
    return int(row_size).bit_length() - 1


def _indices_shape(qid_shape: Tuple[int, ...], indices: Sequence[int]) -> Tuple[int, ...]:
    """Validates that the indices have values within range of `len(qid_shape)`."""
    if any(index < 0 for index in indices):
        raise IndexError(f'Negative index in indices: {indices}')
    if any(index >= len(qid_shape) for index in indices):
        raise IndexError(
            f'Out of range indices, must be less than number of qubits but was {indices}'
        )
    return tuple(qid_shape[i] for i in indices)
