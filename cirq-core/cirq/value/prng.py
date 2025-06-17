# Copyright 2025 The Cirq Developers
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

import numbers
import numpy as np
from typing import Union

from cirq._doc import document


# Type for PRNG or seed-like input.
PRNG_OR_SEED_LIKE = Union[None, int, np.random.RandomState, np.random.Generator]
document(
    PRNG_OR_SEED_LIKE,
    """A pseudorandom number generator or object that can be converted to one.

    Can be an instance of `np.random.Generator`, an integer seed, a `np.random.RandomState`, or None.
    """,
)


def parse_prng(prng_or_seed: PRNG_OR_SEED_LIKE) -> np.random.Generator:
    """Converts the input object into a `numpy.random.Generator`.

    - If `prng_or_seed` is already a `np.random.Generator`, it's returned directly.
    - If `prng_or_seed` is `None`, returns a new `np.random.Generator`
      instance (seeded unpredictably by NumPy).
    - If `prng_or_seed` is an integer, returns `np.random.default_rng(prng_or_seed)`.
    - If `prng_or_seed` is an instance of `np.random.RandomState`, returns a `np.random.Generator` initialized with the RandomState's bit generator or falls back on a random seed.
    - Passing the `np.random` module itself is explicitly disallowed.

    Args:
        prng_or_seed: The object to be used as or converted to a Generator.

    Returns:
        The `numpy.random.Generator` object.

    Raises:
        TypeError: If `prng_or_seed` is the `np.random` module or cannot be
            converted to a `np.random.Generator`.
    """
    if prng_or_seed is np.random:
        raise TypeError(
            "Passing the 'np.random' module is not supported. "
            "Use None to get a default np.random.Generator instance."
        )

    if isinstance(prng_or_seed, np.random.Generator):
        return prng_or_seed

    if prng_or_seed is None:
        return np.random.default_rng()

    if isinstance(prng_or_seed, numbers.Integral):
        return np.random.default_rng(int(prng_or_seed))

    if isinstance(prng_or_seed, np.random.RandomState):
        bit_gen = getattr(prng_or_seed, '_bit_generator', None)
        if bit_gen is not None:
            return np.random.default_rng(bit_gen)
        seed_val = prng_or_seed.randint(2**31)
        return np.random.default_rng(seed_val)

    raise TypeError(
        f"Input {prng_or_seed} (type: {type(prng_or_seed).__name__}) cannot be converted "
        f"to a {np.random.Generator.__name__}"
    )
