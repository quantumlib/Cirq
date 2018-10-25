# Copyright 2018 The Cirq Developers
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

"""Utility methods for transforming matrices."""

from typing import Tuple, Optional, Sequence, List, Union

import numpy as np


def reflection_matrix_pow(reflection_matrix: np.ndarray, exponent: float):
    """Raises a matrix with two opposing eigenvalues to a power.

    Args:
        reflection_matrix: The matrix to raise to a power.
        exponent: The power to raise the matrix to.

    Returns:
        The given matrix raised to the given power.
    """

    # The eigenvalues are x and -x for some complex unit x. Determine x.
    squared_phase = np.dot(reflection_matrix[:, 0],
                           reflection_matrix[0, :])
    phase = complex(np.sqrt(squared_phase))

    # Extract +x and -x eigencomponents of the matrix.
    i = np.eye(reflection_matrix.shape[0]) * phase
    pos_part = (i + reflection_matrix) * 0.5
    neg_part = (i - reflection_matrix) * 0.5

    # Raise the matrix to a power by raising its eigencomponents to that power.
    pos_factor = phase**(exponent - 1)
    neg_factor = pos_factor * complex(-1)**exponent
    pos_part_raised = pos_factor * pos_part
    neg_part_raised = neg_part * neg_factor
    return pos_part_raised + neg_part_raised


def match_global_phase(a: np.ndarray,
                       b: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Phases the given matrices so that they agree on the phase of one entry.

    To maximize precision, the position with the largest entry from one of the
    matrices is used when attempting to compute the phase difference between
    the two matrices.

    Args:
        a: A numpy array.
        b: Another numpy array.

    Returns:
        A tuple (a', b') where a' == b' implies a == b*exp(i t) for some t.
    """

    # Not much point when they have different shapes.
    if a.shape != b.shape:
        return a, b

    # Find the entry with the largest magnitude in one of the matrices.
    k = max(np.ndindex(*a.shape), key=lambda t: abs(b[t]))

    def dephase(v):
        r = np.real(v)
        i = np.imag(v)

        # Avoid introducing floating point error when axis-aligned.
        if i == 0:
            return -1 if r < 0 else 1
        if r == 0:
            return 1j if i < 0 else -1j

        return np.exp(-1j * np.arctan2(i, r))

    # Zero the phase at this entry in both matrices.
    return a * dephase(a[k]), b * dephase(b[k])


def targeted_left_multiply(left_matrix: np.ndarray,
                           right_target: np.ndarray,
                           target_axes: Sequence[int],
                           out: Optional[np.ndarray] = None
                           ) -> np.ndarray:
    """Left-multiplies the given axes of the target tensor by the given matrix.

    Note that the matrix must have a compatible tensor structure.

    For example, if you have an 6-qubit state vector `input_state` with shape
    (2, 2, 2, 2, 2, 2), and a 2-qubit unitary operation `op` with shape
    (2, 2, 2, 2), and you want to apply `op` to the 5'th and 3'rd qubits
    within `input_state`, then the output state vector is computed as follows:

        output_state = cirq.targeted_left_multiply(op, input_state, [5, 3])

    This method also works when the right hand side is a matrix instead of a
    vector. If a unitary circuit's matrix is `old_effect`, and you append
    a CNOT(q1, q4) operation onto the circuit, where the control q1 is the qubit
    at offset 1 and the target q4 is the qubit at offset 4, then the appended
    circuit's unitary matrix is computed as follows:

        new_effect = cirq.targeted_left_multiply(
            left_matrix=cirq.unitary(cirq.CNOT).reshape((2, 2, 2, 2)),
            right_target=old_effect,
            target_axes=[1, 4])

    Args:
        left_matrix: What to left-multiply the target tensor by.
        right_target: A tensor to carefully broadcast a left-multiply over.
        target_axes: Which axes of the target are being operated on.
        out: The buffer to store the results in. If not specified or None, a new
            buffer is used. Must have the same shape as right_target.

    Returns:
        The output tensor.
    """
    k = len(target_axes)
    d = len(right_target.shape)
    work_indices = tuple(range(k))
    data_indices = tuple(range(k, k + d))
    used_data_indices = tuple(data_indices[q] for q in target_axes)
    input_indices = work_indices + used_data_indices
    output_indices = list(data_indices)
    for w, t in zip(work_indices, target_axes):
        output_indices[t] = w

    all_indices = set(input_indices + data_indices + tuple(output_indices))

    return np.einsum(left_matrix, input_indices,
                     right_target, data_indices,
                     output_indices,
                     # We would prefer to omit 'optimize=' (it's faster),
                     # but this is a workaround for a bug in numpy:
                     #     https://github.com/numpy/numpy/issues/10926
                     optimize=len(all_indices) >= 26,
                     # And this is workaround for *another* bug!
                     # Supposed to be able to just say 'old=old'.
                     **({'out': out} if out is not None else {}))


_TSliceAtom = Union[int, slice, 'ellipsis']
_TSlice = Union[_TSliceAtom, Sequence[_TSliceAtom]]


def apply_matrix_to_slices(
        target: np.ndarray,
        matrix: np.ndarray,
        slices: List[_TSlice],
        *,
        out: Optional[np.ndarray] = None) -> np.ndarray:
    """Left-multiplies an NxN matrix onto N slices of a numpy array.

    Example:
        The 4x4 matrix of a fractional SWAP gate can be expressed as

           [ 1       ]
           [   X**t  ]
           [       1 ]

        Where X is the 2x2 Pauli X gate and t is the power of the swap with t=1
        being a full swap. X**t is a power of the Pauli X gate's matrix.
        Applying the fractional swap is equivalent to applying a fractional X
        within the inner 2x2 subspace; the rest of the matrix is identity. This
        can be expressed using `apply_matrix_to_slices` as follows:

            def fractional_swap(target):
                assert target.shape == (4,)
                return apply_matrix_to_slices(
                    target=target,
                    matrix=cirq.unitary(cirq.X**t),
                    slices=[1, 2]
                )

    Args:
        target: The input array with slices that need to be left-multiplied.
        matrix: The linear operation to apply to the subspace defined by the
            slices.
        slices: The parts of the tensor that correspond to the "vector entries"
            that the matrix should operate on. May be integers or complicated
            multi-dimensional slices into a tensor. The slices must refer to
            non-overlapping sections of the input all with the same shape.
        out: Where to write the output. If not specified, a new numpy array is
            created, with the same shape and dtype as the target, to store the
            output.

    Returns:
        The transformed array.
    """
    # Validate arguments.
    if out is target:
        raise ValueError("Can't write output over the input.")
    if matrix.shape != (len(slices), len(slices)):
        raise ValueError("matrix.shape != (len(slices), len(slices))")

    # Fill in default values and prepare space.
    if out is None:
        out = np.copy(target)
    else:
        out[...] = target[...]

    # Apply operation.
    for i, s_i in enumerate(slices):
        out[s_i] *= matrix[i, i]
        for j, s_j in enumerate(slices):
            if i != j:
                out[s_i] += target[s_j] * matrix[i, j]

    return out
