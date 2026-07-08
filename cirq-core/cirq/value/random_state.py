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

from __future__ import annotations

from typing import Any, Union, cast

import numpy as np

from cirq._doc import document

RANDOM_STATE_OR_SEED_LIKE = Union[None, int, np.random.RandomState, np.random.Generator]
document(
    RANDOM_STATE_OR_SEED_LIKE,
    """A pseudorandom number generator or object that can be converted to one.

    If None, turns into the module `np.random`.

    If an integer, turns into a `np.random.RandomState` seeded with that
    integer.

    If a `np.random.RandomState` or `np.random.Generator`, it is used unmodified.
    
    Otherwise, a ValueError is raised.
    """,
)

def parse_random_state(
    random_state: RANDOM_STATE_OR_SEED_LIKE,
) -> Union[np.random.RandomState, np.random.Generator]:
    """Interpret an object as a pseudorandom number generator.

    If `random_state` is None or the `np.random` module, returns the module `np.random`.
    If `random_state` is an integer, returns `np.random.RandomState(random_state)`.
    If `random_state` is a `np.random.RandomState` or `np.random.Generator`, 
    returns it unmodified.
    Otherwise, raises a ValueError.

    Args:
        random_state: The object to be used as or converted to a pseudorandom
            number generator.

    Returns:
        The pseudorandom number generator object.
        
    Raises:
        ValueError: If the random state or seed is not valid.
    """
    # FIX: Explicitly allow the np.random module to pass through
    if random_state is None or random_state is np.random:
        return cast(np.random.RandomState, np.random)
    elif isinstance(random_state, int):
        return np.random.RandomState(random_state)
    elif isinstance(random_state, (np.random.RandomState, np.random.Generator)):
        return random_state
    else:
        raise ValueError(
            f'random_state must be None, an int, a numpy RandomState, a numpy Generator, '
            f'or the numpy.random module. Got {type(random_state)} instead.'
        )