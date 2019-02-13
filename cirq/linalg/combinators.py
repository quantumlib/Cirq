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
from typing import Union, Type

import numpy as np


def kron(*matrices: np.ndarray) -> np.ndarray:
    """Computes the kronecker product of a sequence of matrices.

    A *args version of lambda args: functools.reduce(np.kron, args).

    Args:
        *matrices: The matrices and controls to combine with the kronecker
            product.

    Returns:
        The resulting matrix.
    """
    product = np.eye(1)
    for m in matrices:
        product = np.kron(product, m)
    return np.array(product)


CONTROL_TAG = np.array([[float('nan'), 0], [0, 1]])  # For kron_with_controls


def kron_with_controls(*matrices: np.ndarray) -> np.ndarray:
    """Computes the kronecker product of a sequence of matrices and controls.

    Use linalg.CONTROL_TAG to represent controls. Any entry of the output
    matrix corresponding to a situation where the control is not satisfied will
    be overwritten by identity matrix elements.

    The control logic works by imbuing NaN with the meaning "failed to meet one
    or more controls". The normal kronecker product then spreads the per-item
    NaNs to all the entries in the product that need to be replaced by identity
    matrix elements. This method rewrites those NaNs. Thus CONTROL_TAG can be
    the matrix [[NaN, 0], [0, 1]] or equivalently [[NaN, NaN], [NaN, 1]].

    Because this method re-interprets NaNs as control-failed elements, it won't
    propagate error-indicating NaNs from its input to its output in the way
    you'd otherwise expect.

    Args:
        *matrices: The matrices and controls to combine with the kronecker
            product.

    Returns:
        The resulting matrix.
    """

    product = kron(*matrices)

    # The NaN from CONTROL_TAG spreads to everywhere identity belongs.
    for i in range(product.shape[0]):
        for j in range(product.shape[1]):
            if np.isnan(product[i, j]):
                product[i, j] = 1 if i == j else 0

    return product


def dot(*values: Union[float, complex, np.ndarray]
        ) -> Union[float, complex, np.ndarray]:
    """Computes the dot/matrix product of a sequence of values.

    A *args version of np.linalg.multi_dot.

    Args:
        *values: The values to combine with the dot/matrix product.

    Returns:
        The resulting value or matrix.
    """
    if len(values) == 1:
        if isinstance(values[0], np.ndarray):
            return np.array(values[0])
        return values[0]
    return np.linalg.multi_dot(values)


def _merge_dtypes(dtype1: Type[np.number], dtype2: Type[np.number]
                  ) -> Type[np.number]:
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
