# Copyright 2024 The Cirq Developers
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

from typing import Union

import numbers
import numpy as np

from cirq._doc import document
from cirq.value.random_state import RANDOM_STATE_OR_SEED_LIKE

PRNG_OR_SEED_LIKE = Union[None, int, np.random.RandomState, np.random.Generator]

document(
    PRNG_OR_SEED_LIKE,
    """A pseudorandom number generator or object that can be converted to one.

    If is an integer or None, turns into a `np.random.Generator` seeded with that value.
    If is an instance of `np.random.Generator` or a subclass of it, return as is.
    If is an instance of `np.random.RandomState` or has a `randint` method, returns
    `np.random.default_rng(rs.randint(2**31))`
    """,
)


def parse_prng(
    prng_or_seed: Union[PRNG_OR_SEED_LIKE, RANDOM_STATE_OR_SEED_LIKE]
) -> np.random.Generator:
    """Interpret an object as a pseudorandom number generator.

    If `prng_or_seed` is an `np.random.Generator`, return it unmodified.
    If `prng_or_seed` is None or an integer, returns `np.random.default_rng(prng_or_seed)`.
    If `prng_or_seed` is an instance of `np.random.RandomState` or has a `randint` method,
        returns `np.random.default_rng(prng_or_seed.randint(2**31))`.

    Args:
        prng_or_seed: The object to be used as or converted to a pseudorandom
            number generator.

    Returns:
        The pseudorandom number generator object.

    Raises:
        TypeError: If `prng_or_seed` is can't be converted to an np.random.Generator.
    """
    if isinstance(prng_or_seed, np.random.Generator):
        return prng_or_seed
    if prng_or_seed is None or isinstance(prng_or_seed, numbers.Integral):
        return np.random.default_rng(prng_or_seed if prng_or_seed is None else int(prng_or_seed))
    if isinstance(prng_or_seed, np.random.RandomState):
        return np.random.default_rng(prng_or_seed.randint(2**31))
    randint = getattr(prng_or_seed, "randint", None)
    if randint is not None:
        return np.random.default_rng(randint(2**31))
    raise TypeError(f"{prng_or_seed} can't be converted to a pseudorandom number generator")
