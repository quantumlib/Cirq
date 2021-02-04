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
"""Utility methods for creating vectors and matrices."""

from typing import Sequence, Union, Tuple, Type, Any

import numpy as np

from cirq._compat import deprecated


@deprecated(deadline='v0.9', fix='Use cirq.one_hot instead.')
def one_hot(
    *,
    index: Union[None, int, Sequence[int]] = None,
    shape: Union[int, Sequence[int]],
    value: Any = 1,
    dtype: Type[np.number],
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


@deprecated(deadline='v0.9', fix='Use cirq.eye_tensor instead.')
def eye_tensor(
    half_shape: Tuple[int, ...], *, dtype: Type[np.number]  # Force keyword args
) -> np.array:
    """Returns an identity matrix reshaped into a tensor.

    Args:
        half_shape: A tuple representing the number of quantum levels of each
            qubit the returned matrix applies to.  `half_shape` is (2, 2, 2) for
            a three-qubit identity operation tensor.
        dtype: The numpy dtype of the new array.

    Returns:
        The created numpy array with shape `half_shape + half_shape`.
    """
    state = np.eye(np.prod(half_shape, dtype=int), dtype=dtype)
    state.shape = half_shape * 2
    return state
