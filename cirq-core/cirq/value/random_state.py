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

import numbers
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

PRNG_OR_SEED_LIKE = None | int | np.random.RandomState | np.random.Generator
document(
    PRNG_OR_SEED_LIKE,
    """A pseudorandom number generator or object that can be converted to one.

    If None, turns into a `np.random.Generator`.
    If an integer, turns into a `np.random.Generator` seeded with that value.
    If an instance of `np.random.Generator` or a subclass of it, returns it unmodified.
    If an instance of `np.random.RandomState`, turns into a `np.random.Generator`.
    """,
)


def parse_random_state(random_state: RANDOM_STATE_OR_SEED_LIKE) -> np.random.RandomState:
    """Interpret an object as a pseudorandom number generator.

    If `random_state` is None, returns the module `np.random`.
    If `random_state` is an integer, returns
    `np.random.RandomState(random_state)`.
    Otherwise, returns `random_state` unmodified.

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
    else:
        return cast(np.random.RandomState, random_state)


def parse_random_generator(prng_or_seed: PRNG_OR_SEED_LIKE) -> np.random.Generator:
    """Interpret an object as a pseudorandom number generator.

    If `prng_or_seed` is an `np.random.Generator`, return it unmodified.
    If `prng_or_seed` is None or an integer, returns a new `np.random.Generator`.
    If `prng_or_seed` is an instance of `np.random.RandomState`,
    returns `np.random.default_rng(prng_or_seed._bit_generator)`.

    Args:
        prng_or_seed: The object to be used as or converted to a pseudorandom
            number generator.

    Returns:
        The pseudorandom number generator object.

    Raises:
        TypeError: If `prng_or_seed` can't be converted to an `np.random.Generator`.
    """
    if prng_or_seed is None:
        return np.random.default_rng()
    if isinstance(prng_or_seed, numbers.Integral):
        return np.random.default_rng(int(prng_or_seed))
    if isinstance(prng_or_seed, np.random.Generator):
        return prng_or_seed
    if isinstance(prng_or_seed, np.random.RandomState):
        return np.random.default_rng(prng_or_seed._bit_generator)
    raise TypeError(f"{prng_or_seed} cannot be converted to an np.random.Generator.")
