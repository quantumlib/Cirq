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

import warnings
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

PRNG_OR_SEED_LIKE = Any
document(
    PRNG_OR_SEED_LIKE,
    """A pseudorandom number generator or object that can be converted to one.

    If None, a new `np.random.Generator` is created using the default
    `np.random.default_rng()` (which uses the system entropy).

    If an integer or `np.random.SeedSequence`, a new `np.random.Generator` is
    created using `np.random.default_rng(seed)`.

    If an instance of `np.random.Generator`, it is returned unmodified.

    If an instance of `np.random.RandomState`, a deprecation warning is issued
    and it is returned unmodified (or converted if the context requires a
    Generator, but `RandomState` does not have `Generator` interface, so it
    is kept as is for backward compatibility in this helper, though callers
    should migrate to `Generator`).
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

    Note:
        This function is deprecated. Use `parse_random_generator` instead.
    """
    warnings.warn(
        "parse_random_state is deprecated. Use parse_random_generator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if random_state is None:
        return cast(np.random.RandomState, np.random)
    elif isinstance(random_state, int):
        return np.random.RandomState(random_state)
    else:
        return cast(np.random.RandomState, random_state)


def parse_random_generator(random_state: PRNG_OR_SEED_LIKE) -> np.random.Generator:
    """Interpret an object as a pseudorandom number generator.

    If `random_state` is None, returns a new `np.random.Generator` created
    via `np.random.default_rng()`.
    If `random_state` is an integer or `np.random.SeedSequence`, returns
    `np.random.default_rng(random_state)`.
    If `random_state` is a `np.random.Generator`, returns it unmodified.
    Otherwise, returns `random_state` unmodified.

    Args:
        random_state: The object to be used as or converted to a pseudorandom
            number generator.

    Returns:
        The pseudorandom number generator object.
    """
    if random_state is None:
        return np.random.default_rng()
    elif isinstance(random_state, np.random.Generator):
        return random_state
    elif isinstance(random_state, (int, np.random.SeedSequence)):
        return np.random.default_rng(random_state)
    else:
        return cast(np.random.Generator, random_state)