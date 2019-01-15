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

import numpy as np

DEFAULT_RTOL = 1e-5
DEFAULT_ATOL = 1e-8
DEFAULT_EQUAL_NAN = False


def all_close(a, b, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL,
              equal_nan: bool = DEFAULT_EQUAL_NAN) -> bool:
    """Returns whether the matrices are approximately equal within the tolerance
    parameters.

    Args:
        a: First matrix to compare.
        b: Second matrix to compare.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        equal_nan: Whether to compare NaN's as equal.
    """
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def all_near_zero(a, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL,
                  equal_nan: bool = DEFAULT_EQUAL_NAN) -> bool:
    """Returns whether the matrix approximately contains all zero elements.

    Args:
        a: Matrix to evaluate.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        equal_nan: Whether to compare NaN's as equal.
    """
    return all_close(a, np.zeros(np.shape(a)), rtol, atol, equal_nan)


def all_near_zero_mod(a,
                      period,
                      rtol: float = DEFAULT_RTOL,
                      atol: float = DEFAULT_ATOL,
                      equal_nan: bool = DEFAULT_EQUAL_NAN):
    return all_close((np.array(a) + (period / 2)) % period - period / 2,
                     np.zeros(np.shape(a)), rtol, atol, equal_nan)



def near_zero(a, atol: float = DEFAULT_ATOL):
    return abs(a) <= atol


def near_zero_mod(a, period, atol: float = DEFAULT_ATOL):
    half_period = period / 2
    return near_zero((a + half_period) % period - half_period, atol)
