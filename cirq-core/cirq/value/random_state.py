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

from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from cirq._doc import document

RANDOM_STATE_OR_SEED_LIKE = Any
document(
    RANDOM_STATE_OR_SEED_LIKE,
    """A pseudorandom number generator or object that can be converted to one.

    If None, turns into the module `np.random`.

    If an integer, turns into a `np.random.RandomState` seeded with that
    integer.

    If none of the above, it is used unmodified. In this case, it is assumed
    that the object implements whatever methods are required for the use case
    at hand. For example, it might be an existing instance of
    `np.random.RandomState` or a custom pseudorandom number generator
    implementation.
    """,
)


def parse_random_state(
    random_state: RANDOM_STATE_OR_SEED_LIKE,
) -> np.random.RandomState | np.random.Generator:
    """Interpret an object as a pseudorandom number generator.

    If `random_state` is None, returns the module `np.random`.
    If `random_state` is an integer, returns
    `np.random.RandomState(random_state)`.
    If `random_state` is an `np.random.Generator`, return it unmodified.
    Otherwise, returns `random_state` cast to an `np.random.RandomState`.

    Args:
        random_state: The object to be used as or converted to a pseudorandom
            number generator.

    Returns:
        The pseudorandom number generator object.
    """
    if random_state is None:
        return cast(np.random.RandomState, np.random)
    elif isinstance(random_state, int):
        return np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.Generator):
        return random_state
    else:
        return cast(np.random.RandomState, random_state)


def get_random_array(
    rng: np.random.RandomState | np.random.Generator, shape: Sequence[int] | None = None
):
    if isinstance(rng, np.random.Generator):
        if shape is not None:
            return rng.random(size=shape)
        else:
            return rng.random()
    else:
        if shape is not None:
            return rng.rand(*shape)
        else:
            return rng.rand()


def get_random_normal_array(
    rng: np.random.RandomState | np.random.Generator, shape: Sequence[int] | None = None
):
    if isinstance(rng, np.random.Generator):
        if shape is not None:
            return rng.standard_normal(size=shape)
        else:
            return rng.standard_normal()
    else:
        if shape is not None:
            return rng.randn(*shape)
        else:
            return rng.randn()


def get_random_int(
    rng: np.random.RandomState | np.random.Generator,
    low: int,
    high: int | None = None,
    size: Sequence[int] | None = None,
):
    if isinstance(rng, np.random.Generator):
        return rng.integers(low, high, size=size)
    else:
        return rng.randint(low, high, size=size)
