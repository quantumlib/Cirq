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

from typing import Sequence, Union, Type

import numpy as np


def one_hot(*,
            index: Union[None, int, Sequence[int]] = None,
            shape: Union[int, Sequence[int]],
            dtype: Type[np.number]) -> np.ndarray:
    """Returns a numpy array with a single 1 entry, and 0 everywhere else.

    Args:
        index: The index that should store 1 instead of 0.
            If not specified, defaults to the start of the array.
        shape: The shape of the array.
        dtype: The dtype of the array.

    Returns:
        The created numpy array.
    """
    if index is None:
        index = 0 if isinstance(shape, int) else (0,) * len(shape)
    result = np.zeros(shape=shape, dtype=dtype)
    result[index] = 1
    return result
