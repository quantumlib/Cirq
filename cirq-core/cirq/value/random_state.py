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

from typing import cast

import numpy as np

from cirq._doc import document

RANDOM_STATE_OR_SEED_LIKE = None | int | np.integer | np.random.RandomState | np.random.Generator
document(
    RANDOM_STATE_OR_SEED_LIKE,
    """A pseudorandom number generator or object that can be converted to one.

    If None, turns into a new `np.random.Generator` spawned from Cirq's default generator.
    If an integer or NumPy integer, turns into a `np.random.Generator` seeded
    with that value.
    If an instance of `np.random.Generator`, it is used unmodified.
    If an instance of `np.random.RandomState`, it is wrapped in a
    `np.random.Generator` that shares its underlying bit generator.
    """,
)

_DEFAULT_RNG: np.random.Generator = np.random.default_rng()


def reseed_default_prng(seed: int) -> None:
    """Reseeds Cirq's default pseudorandom number generator in place.

    Args:
        seed: The seed for the new default generator.
    """
    global _DEFAULT_RNG
    with _DEFAULT_RNG.bit_generator.lock:
        _DEFAULT_RNG.bit_generator.state = np.random.default_rng(seed).bit_generator.state


def parse_random_state(random_state: RANDOM_STATE_OR_SEED_LIKE) -> np.random.RandomState:
    """Interpret an object as a pseudorandom number generator.

    If `random_state` is None, returns the module `np.random`.
    If `random_state` is an integer, returns
    `np.random.RandomState(random_state)`.
    If `random_state` is an `np.random.Generator`, returns a `np.random.RandomState`
    that shares its underlying bit generator.
    Otherwise, returns `random_state` unmodified.

    Args:
        random_state: The object to be used as or converted to a pseudorandom
            number generator.

    Returns:
        The pseudorandom number generator object.
    """
    if random_state is None:
        return cast(np.random.RandomState, np.random)
    elif isinstance(random_state, (int, np.integer)):
        return np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.Generator):
        return np.random.RandomState(random_state.bit_generator)
    else:
        return random_state


def parse_random_generator(prng_or_seed: RANDOM_STATE_OR_SEED_LIKE) -> np.random.Generator:
    """Interpret an object as a pseudorandom number generator.

    If `prng_or_seed` is an `np.random.Generator`, return it unmodified.
    If `prng_or_seed` is None, returns Cirq's default `np.random.Generator`.
    If `prng_or_seed` is an integer or NumPy integer, returns a new `np.random.Generator` seeded
    with that value.
    If `prng_or_seed` is an instance of `np.random.RandomState`,
    returns `np.random.default_rng(prng_or_seed)`.

    Args:
        prng_or_seed: The object to be used as or converted to a pseudorandom
            number generator.

    Returns:
        The pseudorandom number generator object.

    Raises:
        TypeError: If `prng_or_seed` can't be converted to an `np.random.Generator`.
    """
    if prng_or_seed is None:
        return _DEFAULT_RNG
    if isinstance(prng_or_seed, (int, np.integer)):
        return np.random.default_rng(prng_or_seed)
    if isinstance(prng_or_seed, np.random.Generator):
        return prng_or_seed
    if isinstance(prng_or_seed, np.random.RandomState):
        return np.random.default_rng(prng_or_seed)
    raise TypeError(f"{prng_or_seed} cannot be converted to an np.random.Generator.")
