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
"""Utility methods for checking properties of matrices."""
from typing import cast, List, Optional, Sequence, Union, Tuple

import numpy as np

from cirq.linalg import tolerance, transformations
from cirq import value


def is_diagonal(matrix: np.ndarray, *, atol: float = 1e-8) -> np.bool_:
    """Determines if a matrix is a approximately diagonal.

    A matrix is diagonal if i!=j implies m[i,j]==0.

    Args:
        matrix: The matrix to check.
        atol: The per-matrix-entry absolute tolerance on equality.

    Returns:
        Whether the matrix is diagonal within the given tolerance.
    """
    matrix = np.copy(matrix)
    for i in range(min(matrix.shape)):
        matrix[i, i] = 0
    return tolerance.all_near_zero(matrix, atol=atol)


def is_hermitian(matrix: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Determines if a matrix is approximately Hermitian.

    A matrix is Hermitian if it's square and equal to its adjoint.

    Args:
        matrix: The matrix to check.
        rtol: The per-matrix-entry relative tolerance on equality.
        atol: The per-matrix-entry absolute tolerance on equality.

    Returns:
        Whether the matrix is Hermitian within the given tolerance.
    """
    return matrix.shape[0] == matrix.shape[1] and np.allclose(
        matrix, np.conj(matrix.T), rtol=rtol, atol=atol
    )


def is_orthogonal(matrix: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Determines if a matrix is approximately orthogonal.

    A matrix is orthogonal if it's square and real and its transpose is its
    inverse.

    Args:
        matrix: The matrix to check.
        rtol: The per-matrix-entry relative tolerance on equality.
        atol: The per-matrix-entry absolute tolerance on equality.

    Returns:
        Whether the matrix is orthogonal within the given tolerance.
    """
    return (
        matrix.shape[0] == matrix.shape[1]
        and np.all(np.imag(matrix) == 0).item()
        and np.allclose(matrix.dot(matrix.T), np.eye(matrix.shape[0]), rtol=rtol, atol=atol)
    )


def is_special_orthogonal(matrix: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Determines if a matrix is approximately special orthogonal.

    A matrix is special orthogonal if it is square and real and its transpose
    is its inverse and its determinant is one.

    Args:
        matrix: The matrix to check.
        rtol: The per-matrix-entry relative tolerance on equality.
        atol: The per-matrix-entry absolute tolerance on equality.

    Returns:
        Whether the matrix is special orthogonal within the given tolerance.
    """
    return is_orthogonal(matrix, rtol=rtol, atol=atol) and (
        matrix.shape[0] == 0 or np.allclose(np.linalg.det(matrix), 1, rtol=rtol, atol=atol)
    )


def is_unitary(matrix: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Determines if a matrix is approximately unitary.

    A matrix is unitary if it's square and its adjoint is its inverse.

    Args:
        matrix: The matrix to check.
        rtol: The per-matrix-entry relative tolerance on equality.
        atol: The per-matrix-entry absolute tolerance on equality.

    Returns:
        Whether the matrix is unitary within the given tolerance.
    """
    return matrix.shape[0] == matrix.shape[1] and np.allclose(
        matrix.dot(np.conj(matrix.T)), np.eye(matrix.shape[0]), rtol=rtol, atol=atol
    )


def is_special_unitary(matrix: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Determines if a matrix is approximately unitary with unit determinant.

    A matrix is special-unitary if it is square and its adjoint is its inverse
    and its determinant is one.

    Args:
        matrix: The matrix to check.
        rtol: The per-matrix-entry relative tolerance on equality.
        atol: The per-matrix-entry absolute tolerance on equality.
    Returns:
        Whether the matrix is unitary with unit determinant within the given
        tolerance.
    """
    return is_unitary(matrix, rtol=rtol, atol=atol) and (
        matrix.shape[0] == 0 or np.allclose(np.linalg.det(matrix), 1, rtol=rtol, atol=atol)
    )


def is_normal(matrix: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Determines if a matrix is approximately normal.

    A matrix is normal if it's square and commutes with its adjoint.

    Args:
        matrix: The matrix to check.
        rtol: The per-matrix-entry relative tolerance on equality.
        atol: The per-matrix-entry absolute tolerance on equality.

    Returns:
        Whether the matrix is normal within the given tolerance.
    """
    return matrix_commutes(matrix, matrix.T.conj(), rtol=rtol, atol=atol)


def is_cptp(*, kraus_ops: Sequence[np.ndarray], rtol: float = 1e-5, atol: float = 1e-8):
    """Determines if a channel is completely positive trace preserving (CPTP).

    A channel composed of Kraus operators K[0:n] is a CPTP map if the sum of
    the products `adjoint(K[i]) * K[i])` is equal to 1.

    Args:
        kraus_ops: The Kraus operators of the channel to check.
        rtol: The relative tolerance on equality.
        atol: The absolute tolerance on equality.
    """
    sum_ndarray = cast(np.ndarray, sum(matrix.T.conj() @ matrix for matrix in kraus_ops))
    return np.allclose(sum_ndarray, np.eye(*sum_ndarray.shape), rtol=rtol, atol=atol)


def matrix_commutes(
    m1: np.ndarray, m2: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
    """Determines if two matrices approximately commute.

    Two matrices A and B commute if they are square and have the same size and
    AB = BA.

    Args:
        m1: One of the matrices.
        m2: The other matrix.
        rtol: The per-matrix-entry relative tolerance on equality.
        atol: The per-matrix-entry absolute tolerance on equality.

    Returns:
        Whether the two matrices have compatible sizes and a commutator equal
        to zero within tolerance.
    """
    return (
        m1.shape[0] == m1.shape[1]
        and m1.shape == m2.shape
        and np.allclose(m1.dot(m2), m2.dot(m1), rtol=rtol, atol=atol)
    )


def allclose_up_to_global_phase(
    a: np.ndarray,
    b: np.ndarray,
    *,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    equal_nan: bool = False,
) -> bool:
    """Determines if a ~= b * exp(i t) for some t.

    Args:
        a: A numpy array.
        b: Another numpy array.
        rtol: Relative error tolerance.
        atol: Absolute error tolerance.
        equal_nan: Whether or not NaN entries should be considered equal to
            other NaN entries.
    """

    if a.shape != b.shape:
        return False
    a, b = transformations.match_global_phase(a, b)

    # Should now be equivalent.
    return np.allclose(a=a, b=b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def slice_for_qubits_equal_to(
    target_qubit_axes: Sequence[int],
    little_endian_qureg_value: int = 0,
    *,  # Forces keyword args.
    big_endian_qureg_value: int = 0,
    num_qubits: Optional[int] = None,
    qid_shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[Union[slice, int, 'ellipsis'], ...]:
    """Returns an index corresponding to a desired subset of an np.ndarray.

    It is assumed that the np.ndarray's shape is of the form (2, 2, 2, ..., 2).

    Example:
        ```python
        # A '4 qubit' tensor with values from 0 to 15.
        r = np.array(range(16)).reshape((2,) * 4)

        # We want to index into the subset where qubit #1 and qubit #3 are ON.
        s = cirq.slice_for_qubits_equal_to([1, 3], 0b11)
        print(s)
        # (slice(None, None, None), 1, slice(None, None, None), 1, Ellipsis)

        # Get that subset. It corresponds to numbers of the form 0b*1*1.
        # where here '*' indicates any possible value.
        print(r[s])
        # [[ 5  7]
        #  [13 15]]
        ```

    Args:
        target_qubit_axes: The qubits that are specified by the index bits. All
            other axes of the slice are unconstrained.
        little_endian_qureg_value: An integer whose bits specify what value is
            desired for of the target qubits. The integer is little endian
            w.r.t. the target qubit axes, meaning the low bit of the integer
            determines the desired value of the first targeted qubit, and so
            forth with the k'th targeted qubit's value set to
            bool(qureg_value & (1 << k)).
        big_endian_qureg_value: Same as `little_endian_qureg_value` but big
            endian w.r.t. to target qubit axes, meaning the low bit of the
            integer dertemines the desired value of the last target qubit, and
            so forth.  Specify exactly one of the `*_qureg_value` arguments.
        num_qubits: If specified the slices will extend all the way up to
            this number of qubits, otherwise if it is None, the final element
            return will be Ellipsis. Optional and defaults to using Ellipsis.
        qid_shape: The qid shape of the state vector being sliced.  Specify this
            instead of `num_qubits` when using qids with dimension != 2.  The
            qureg value is interpreted to store digits with corresponding bases
            packed into an int.

    Returns:
        An index object that will slice out a mutable view of the desired subset
        of a tensor.

    Raises:
        ValueError: If the `qid_shape` mismatches `num_qubits` or exactly one of
            `little_endian_qureg_value` and `big_endian_qureg_value` is not
            specified.
    """
    qid_shape_specified = qid_shape is not None
    if qid_shape is not None or num_qubits is not None:
        if num_qubits is None:
            num_qubits = len(cast(Tuple[int, ...], qid_shape))
        elif qid_shape is None:
            qid_shape = (2,) * num_qubits
        if num_qubits != len(cast(Tuple[int, ...], qid_shape)):
            raise ValueError('len(qid_shape) != num_qubits')
    if little_endian_qureg_value and big_endian_qureg_value:
        raise ValueError(
            'Specify exactly one of the arguments little_endian_qureg_value '
            'or big_endian_qureg_value.'
        )
    out_size_specified = num_qubits is not None
    out_size = (
        cast(int, num_qubits) if out_size_specified else max(target_qubit_axes, default=-1) + 1
    )
    result = cast(List[Union[slice, int, 'ellipsis']], [slice(None)] * out_size)
    if not out_size_specified:
        result.append(Ellipsis)
    if qid_shape is None:
        qid_shape = (2,) * out_size
    target_shape = tuple(qid_shape[i] for i in target_qubit_axes)
    if big_endian_qureg_value:
        digits = value.big_endian_int_to_digits(big_endian_qureg_value, base=target_shape)
    else:
        if little_endian_qureg_value < 0 and not qid_shape_specified:
            # Allow negative binary numbers
            little_endian_qureg_value &= (1 << len(target_shape)) - 1
        digits = value.big_endian_int_to_digits(little_endian_qureg_value, base=target_shape[::-1])[
            ::-1
        ]
    for axis, digit in zip(target_qubit_axes, digits):
        result[axis] = digit
    return tuple(result)
