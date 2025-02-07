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

"""Utility methods for combining matrices."""

import functools
from typing import Union, TYPE_CHECKING

import numpy as np

from cirq._doc import document

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, ArrayLike


def kron(*factors: Union[np.ndarray, complex], shape_len: int = 2) -> np.ndarray:
    """Computes the kronecker product of a sequence of values.

    A *args version of lambda args: functools.reduce(np.kron, args).

    Args:
        *factors: The matrices, tensors, and/or scalars to combine together
            using np.kron.
        shape_len: The expected number of dimensions in the output. Mainly
            determines the behavior of the empty kron product.

    Returns:
        The kronecker product of all the inputs.
    """
    product: np.ndarray = np.ones(shape=(1,) * shape_len)
    for m in factors:
        product = np.kron(product, m)
    return np.array(product)


CONTROL_TAG = np.array([[float('nan'), 0], [0, 1]])
document(
    CONTROL_TAG,
    """A special indicator value for `cirq.kron_with_controls`.

    This value is a stand-in for "control operations on the other qubits based
    on the value of this qubit", which otherwise doesn't have a proper matrix.
    """,
)


def kron_with_controls(*factors: Union[np.ndarray, complex]) -> np.ndarray:
    """Computes the kronecker product of a sequence of values and control tags.

    Use `cirq.CONTROL_TAG` to represent controls. Any entry of the output
    corresponding to a situation where the control is not satisfied will
    be overwritten by identity matrix elements.

    The control logic works by imbuing NaN with the meaning "failed to meet one
    or more controls". The normal kronecker product then spreads the per-item
    NaNs to all the entries in the product that need to be replaced by identity
    matrix elements. This method rewrites those NaNs. Thus CONTROL_TAG can be
    the matrix [[NaN, 0], [0, 1]] or equivalently [[NaN, NaN], [NaN, 1]].

    Because this method re-interprets NaNs as control-failed elements, it won't
    propagate error-indicating NaNs from its input to its output in the way
    you'd otherwise expect.

    Examples:

        ```
        result = cirq.kron_with_controls(
            cirq.CONTROL_TAG,
            cirq.unitary(cirq.X))
        print(result.astype(np.int32))

        # prints:
        # [[1 0 0 0]
        #  [0 1 0 0]
        #  [0 0 0 1]
        #  [0 0 1 0]]
        ```

    Args:
        *factors: The matrices, tensors, scalars, and/or control tags to combine
            together using np.kron.

    Returns:
        The resulting matrix.
    """

    product = kron(*factors)

    # The NaN from CONTROL_TAG spreads to everywhere identity belongs.
    for i in range(product.shape[0]):
        for j in range(product.shape[1]):
            if np.isnan(product[i, j]):
                product[i, j] = 1 if i == j else 0

    return product


def dot(*values: 'ArrayLike') -> np.ndarray:
    """Computes the dot/matrix product of a sequence of values.

    Performs the computation in serial order without regard to the matrix
    sizes.  If you are using this for matrices of large and differing sizes,
    consider using np.lingalg.multi_dot for better performance.

    Args:
        *values: The values to combine with the dot/matrix product.

    Returns:
        The resulting value or matrix.

    Raises:
        ValueError: If the method is called without any arguments.
    """
    if len(values) == 0:
        raise ValueError("cirq.dot must be called with arguments")

    if len(values) == 1:
        # note: it's important that we copy input arrays.
        return np.array(values[0])

    result = np.asarray(values[0])
    for value in values[1:]:
        result = np.dot(result, value)
    return result


def _merge_dtypes(dtype1: 'DTypeLike', dtype2: 'DTypeLike') -> np.dtype:
    return (np.zeros(0, dtype1) + np.zeros(0, dtype2)).dtype


def block_diag(*blocks: np.ndarray) -> np.ndarray:
    """Concatenates blocks into a block diagonal matrix.

    Args:
        *blocks: Square matrices to place along the diagonal of the result.

    Returns:
        A block diagonal matrix with the given blocks along its diagonal.

    Raises:
        ValueError: A block isn't square.
    """
    for b in blocks:
        if b.shape[0] != b.shape[1]:
            raise ValueError('Blocks must be square.')

    if not blocks:
        return np.zeros((0, 0), dtype=np.complex128)

    n = sum(b.shape[0] for b in blocks)
    dtype = functools.reduce(_merge_dtypes, (b.dtype for b in blocks))

    result = np.zeros(shape=(n, n), dtype=dtype)
    i = 0
    for b in blocks:
        j = i + b.shape[0]
        result[i:j, i:j] = b
        i = j

    return result
