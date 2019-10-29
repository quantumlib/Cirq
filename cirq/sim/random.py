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
"""Utilities for helping with randomness in simulations."""

from typing import Optional, Union

import numpy as np


def prng_from_seed(seed: Optional[Union[int, np.random.RandomState]]):
    """Return a numpy pseudo-random number generator.

    Args:
        seed: If None, will use `np.random`. If an instance of
            `np.random.RandomState`, returns this. If an integer, this will use
            `np.RandomState` seeded by this integer.

    Raises:
        ValueError if the type supplied is not correct.
    """
    if seed is None:
        return np.random
    elif seed == np.random or isinstance(seed, np.random.RandomState):
        return seed
    elif isinstance(seed, int):
        return np.random.RandomState(seed)
    else:
        raise ValueError('Pseudo-random number seed can only be an integer, '
                         'None, or an numpy RandomState. Instead was {}'.format(
                             type(seed)))
