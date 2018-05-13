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

"""Utility methods for transforming matrices."""

from typing import Tuple

import numpy as np


def reflection_matrix_pow(reflection_matrix: np.ndarray, exponent: float):
    """Raises a matrix with two opposing eigenvalues to a power.

    Args:
        reflection_matrix: The matrix to raise to a power.
        exponent: The power to raise the matrix to.

    Returns:
        The given matrix raised to the given power.
    """

    # The eigenvalues are x and -x for some complex unit x. Determine x.
    squared_phase = np.dot(reflection_matrix[:, 0],
                           reflection_matrix[0, :])
    phase = complex(np.sqrt(squared_phase))

    # Extract +x and -x eigencomponents of the matrix.
    i = np.eye(reflection_matrix.shape[0]) * phase
    pos_part = (i + reflection_matrix) * 0.5
    neg_part = (i - reflection_matrix) * 0.5

    # Raise the matrix to a power by raising its eigencomponents to that power.
    pos_factor = phase**(exponent - 1)
    neg_factor = pos_factor * complex(-1)**exponent
    pos_part_raised = pos_factor * pos_part
    neg_part_raised = neg_part * neg_factor
    return pos_part_raised + neg_part_raised


def match_global_phase(a: np.ndarray,
                       b: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Phases the given matrices so that they agree on the phase of one entry.

    To maximize precision, the position with the largest entry from one of the
    matrices is used when attempting to compute the phase difference between
    the two matrices.

    Args:
        a: A numpy array.
        b: Another numpy array.

    Returns:
        A tuple (a', b') where a' == b' implies a == b*exp(i t) for some t.
    """

    # Not much point when they have different shapes.
    if a.shape != b.shape:
        return a, b

    # Find the entry with the largest magnitude in one of the matrices.
    k = max(np.ndindex(*a.shape), key=lambda t: abs(b[t]))

    def dephase(v):
        r = np.real(v)
        i = np.imag(v)

        # Avoid introducing floating point error when axis-aligned.
        if i == 0:
            return -1 if r < 0 else 1
        if r == 0:
            return 1j if i < 0 else -1j

        return np.exp(-1j * np.arctan2(i, r))

    # Zero the phase at this entry in both matrices.
    return a * dephase(a[k]), b * dephase(b[k])
