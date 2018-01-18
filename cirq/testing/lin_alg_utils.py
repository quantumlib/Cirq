# Copyright 2017 Google LLC
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

import numpy as np


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


def allclose_up_to_global_phase(actual: np.ndarray,
                                desired: np.ndarray,
                                atol: float = 1e-8):
    n = desired.shape[0]

    # Find the entry with the largest magnitude in the desired matrix.
    k = max(((i, j) for i in range(n) for j in range(n)),
            key=lambda t: abs(desired[t]))
    dephase_actual = abs(actual[k]) / actual[k] if actual[k] else 1
    dephase_desired = abs(desired[k]) / desired[k] if desired[k] else 1

    # Zero the phase at this entry in both matrices.
    actual_corrected = actual * dephase_actual
    desired_corrected = desired * dephase_desired

    # Should now be equivalent.
    return np.allclose(actual_corrected, desired_corrected, atol=atol)
