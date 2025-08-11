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

"""Utilities for handling probabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cirq.qis import to_valid_state_vector

if TYPE_CHECKING:
    import cirq


def validate_probability(p: float, p_str: str) -> float:
    """Validates that a probability is between 0 and 1 inclusively.

    Args:
        p: The value to validate.
        p_str: What to call the probability in error messages.

    Returns:
        The probability p if the probability if valid.

    Raises:
        ValueError: If the probability is invalid.
    """
    if p < 0:
        raise ValueError(f'{p_str} was less than 0.')
    elif p > 1:
        raise ValueError(f'{p_str} was greater than 1.')
    return p


def state_vector_to_probabilities(state_vector: cirq.STATE_VECTOR_LIKE) -> np.ndarray:
    """Function to transform a state vector like object into a numpy array of probabilities."""
    valid_state_vector = to_valid_state_vector(state_vector)
    return np.abs(valid_state_vector) ** 2
