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
from typing import Sequence, Union, Tuple, TYPE_CHECKING

import numpy as np

from cirq.linalg.tolerance import Tolerance
from cirq.linalg.transformations import match_global_phase

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import List


def is_diagonal(
        matrix: np.ndarray,
        tolerance: Tolerance = Tolerance.DEFAULT
) -> bool:
    """Determines if a matrix is a approximately diagonal.

    A matrix is diagonal if i!=j implies m[i,j]==0.

    Args:
        matrix: The matrix to check.
        tolerance: The per-matrix-entry tolerance on equality.

    Returns:
        Whether the matrix is diagonal within the given tolerance.
    """
    matrix = np.copy(matrix)
    for i in range(min(matrix.shape)):
        matrix[i, i] = 0
    return tolerance.all_near_zero(matrix)


def is_hermitian(
        matrix: np.ndarray,
        tolerance: Tolerance = Tolerance.DEFAULT
) -> bool:
    """Determines if a matrix is approximately Hermitian.

    A matrix is Hermitian if it's square and equal to its adjoint.

    Args:
        matrix: The matrix to check.
        tolerance: The per-matrix-entry tolerance on equality.

    Returns:
        Whether the matrix is Hermitian within the given tolerance.
    """
    return (matrix.shape[0] == matrix.shape[1] and
            tolerance.all_close(matrix, np.conj(matrix.T)))


def is_orthogonal(
        matrix: np.ndarray,
        tolerance: Tolerance = Tolerance.DEFAULT
) -> bool:
    """Determines if a matrix is approximately orthogonal.

    A matrix is orthogonal if it's square and real and its transpose is its
    inverse.

    Args:
        matrix: The matrix to check.
        tolerance: The per-matrix-entry tolerance on equality.

    Returns:
        Whether the matrix is orthogonal within the given tolerance.
    """
    return (matrix.shape[0] == matrix.shape[1] and
            np.all(np.imag(matrix) == 0) and
            tolerance.all_close(matrix.dot(matrix.T), np.eye(matrix.shape[0])))


def is_special_orthogonal(
        matrix: np.ndarray,
        tolerance: Tolerance = Tolerance.DEFAULT
) -> bool:
    """Determines if a matrix is approximately special orthogonal.

    A matrix is special orthogonal if it is square and real and its transpose
    is its inverse and its determinant is one.

    Args:
        matrix: The matrix to check.
        tolerance: The per-matrix-entry tolerance on equality.

    Returns:
        Whether the matrix is special orthogonal within the given tolerance.
    """
    return (is_orthogonal(matrix, tolerance) and
            (matrix.shape[0] == 0 or
             tolerance.all_close(np.linalg.det(matrix), 1)))


def is_unitary(
        matrix: np.ndarray,
        tolerance: Tolerance = Tolerance.DEFAULT
) -> bool:
    """Determines if a matrix is approximately unitary.

    A matrix is unitary if it's square and its adjoint is its inverse.

    Args:
        matrix: The matrix to check.
        tolerance: The per-matrix-entry tolerance on equality.

    Returns:
        Whether the matrix is unitary within the given tolerance.
    """
    return (matrix.shape[0] == matrix.shape[1] and tolerance.all_close(
        matrix.dot(np.conj(matrix.T)), np.eye(matrix.shape[0])))


def is_special_unitary(
        matrix: np.ndarray,
        tolerance: Tolerance = Tolerance.DEFAULT
) -> bool:
    """Determines if a matrix is approximately unitary with unit determinant.

    A matrix is special-unitary if it is square and its adjoint is its inverse
    and its determinant is one.

    Args:
        matrix: The matrix to check.
        tolerance: The per-matrix-entry tolerance on equality.

    Returns:
        Whether the matrix is unitary with unit determinant within the given
        tolerance.
    """
    return (is_unitary(matrix, tolerance) and
            (matrix.shape[0] == 0 or
             tolerance.all_close(np.linalg.det(matrix), 1)))


def commutes(
        m1: np.ndarray,
        m2: np.ndarray,
        tolerance: Tolerance = Tolerance.DEFAULT
) -> bool:
    """Determines if two matrices approximately commute.

    Two matrices A and B commute if they are square and have the same size and
    AB = BA.

    Args:
        m1: One of the matrices.
        m2: The other matrix.
        tolerance: The per-matrix-entry tolerance on equality.

    Returns:
        Whether the two matrices have compatible sizes and a commutator equal
        to zero within tolerance.
  """
    return (m1.shape[0] == m1.shape[1] and
            m1.shape == m2.shape and
            tolerance.all_close(m1.dot(m2), m2.dot(m1)))


def allclose_up_to_global_phase(
        a: np.ndarray,
        b: np.ndarray,
        rtol: float = 1.e-5,
        atol: float = 1.e-8,
        equal_nan: bool = False
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

    a, b = match_global_phase(a, b)

    # Should now be equivalent.
    return np.allclose(a=a, b=b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def slice_for_qubits_equal_to(target_qubit_axes: Sequence[int],
                              little_endian_qureg_value: int,
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
            w.r.t. the target quit axes, meaning the low bit of the integer
            determines the desired value of the first targeted qubit, and so
            forth with the k'th targeted qubit's value set to
            bool(qureg_value & (1 << k)).

    Returns:
        An index object that will slice out a mutable view of the desired subset
        of a tensor.
    """
    n = max(target_qubit_axes) if target_qubit_axes else -1
    result = [slice(None)] * (n + 2)  # type: List[Union[slice, int, ellipsis]]
    for k, axis in enumerate(target_qubit_axes):
        result[axis] = (little_endian_qureg_value >> k) & 1
    result[-1] = Ellipsis
    return tuple(result)
