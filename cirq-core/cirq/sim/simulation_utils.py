# Copyright 2023 The Cirq Developers
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
from typing import Sequence, Tuple

import numpy as np

from cirq import linalg


def state_probabilities_by_indices(
    state_probability: np.ndarray, indices: Sequence[int], qid_shape: Tuple[int, ...]
) -> np.ndarray:
    """Returns the probabilities for a state/measurement on the given indices.

    Args:
        state_probability: The multi-qubit state probability vector. This is an
            array of 2 to the power of the number of real numbers, and
            so state must be of size ``2**integer``.  The `state_probability` can be
            a vector of size ``2**integer`` or a tensor of shape
            ``(2, 2, ..., 2)``.
        indices: Which qubits are measured. The `state_probability` is assumed to be
            supplied in big endian order. That is the xth index of v, when
            expressed as a bitstring, has its largest values in the 0th index.
        qid_shape: The qid shape of the `state_probability`.

    Returns:
        State probabilities.
    """
    probs = state_probability.reshape((-1,))
    not_measured = [i for i in range(len(qid_shape)) if i not in indices]
    if linalg.can_numpy_support_shape(qid_shape):
        # Use numpy transpose if we can since it's more efficient.
        probs = probs.reshape(qid_shape)
        probs = np.transpose(probs, list(indices) + not_measured)
        probs = probs.reshape((-1,))
    else:
        # If we can't use numpy due to numpy/numpy#5744, use a slower method.
        probs = linalg.transpose_flattened_array(probs, qid_shape, list(indices) + not_measured)

    if len(not_measured):
        # Not all qudits are measured.
        volume = np.prod([qid_shape[i] for i in indices])
        # Reshape into a 2D array in which each of the measured states correspond to a row.
        probs = probs.reshape((volume, -1))
        probs = np.sum(probs, axis=-1)

    # To deal with rounding issues, ensure that the probabilities sum to 1.
    return probs / np.sum(probs)
