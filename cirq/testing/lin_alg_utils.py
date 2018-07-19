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

"""A testing class with utilities for checking linear algebra."""

from typing import Optional

import numpy as np

from cirq import linalg


def random_unitary(dim: int) -> np.ndarray:
    """Returns a random unitary matrix distributed with Haar measure.

    Args:
      dim: The width and height of the matrix.

    Returns:
      The sampled unitary matrix.

    References:
        'How to generate random matrices from the classical compact groups'
        http://arxiv.org/abs/math-ph/0609050
    """
    z = (np.random.randn(dim, dim) +
         1j * np.random.randn(dim, dim)) * np.sqrt(0.5)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * (d / abs(d))


def random_orthogonal(dim: int) -> np.ndarray:
    # TODO(craiggidney): Distribute with Haar measure.
    m = np.random.randn(dim, dim) * 2 - 1
    q, _ = np.linalg.qr(m)
    return q


def random_special_unitary(dim: int) -> np.ndarray:
    r = random_unitary(dim)
    r[0, :] /= np.linalg.det(r)
    return r


def random_special_orthogonal(dim: int) -> np.ndarray:
    m = random_orthogonal(dim)
    if np.linalg.det(m) < 0:
        m[0, :] *= -1
    return m


def assert_allclose_up_to_global_phase(
        actual: np.ndarray,
        desired: np.ndarray,
        *,  # Forces keyword args.
        rtol: float = 1e-7,
        atol: float,  # Require atol to be specified
        equal_nan: bool = True,
        err_msg: Optional[str] = '',
        verbose: bool = True) -> None:
    """Checks if a ~= b * exp(i t) for some t.

    Args:
        actual: A numpy array.
        desired: Another numpy array.
        rtol: Relative error tolerance.
        atol: Absolute error tolerance.
        equal_nan: Whether or not NaN entries should be considered equal to
            other NaN entries.
        err_msg: The error message to be printed in case of failure.
        verbose: If True, the conflicting values are appended to the error
            message.

    Raises:
        AssertionError: The matrices aren't nearly equal up to global phase.
    """
    actual, desired = linalg.match_global_phase(actual, desired)
    np.testing.assert_allclose(
        actual=actual,
        desired=desired,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        err_msg=err_msg,
        verbose=verbose)
