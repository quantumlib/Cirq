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

"""Utility for testing approximate equality of matrices and scalars within
tolerances."""
from typing import Union, Iterable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def all_near_zero(a: 'ArrayLike', *, atol: float = 1e-8) -> np.bool_:
    """Checks if the tensor's elements are all near zero.

    Args:
        a: Tensor of elements that could all be near zero.
        atol: Absolute tolerance.
    """
    return np.all(np.less_equal(np.abs(a), atol))


def all_near_zero_mod(
    a: Union[float, complex, Iterable[float], np.ndarray], period: float, *, atol: float = 1e-8
) -> np.bool_:
    """Checks if the tensor's elements are all near multiples of the period.

    Args:
        a: Tensor of elements that could all be near multiples of the period.
        period: The period, e.g. 2 pi when working in radians.
        atol: Absolute tolerance.
    """
    b = (np.asarray(a) + period / 2) % period - period / 2
    return np.all(np.less_equal(np.abs(b), atol))


def near_zero(a: float, *, atol: float = 1e-8) -> bool:
    return abs(a) <= atol


def near_zero_mod(a: float, period: float, *, atol: float = 1e-8) -> bool:
    half_period = period / 2
    return near_zero((a + half_period) % period - half_period, atol=atol)
