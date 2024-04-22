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

import abc
from typing import TypeVar, Union, overload

import numpy as np

from cirq._doc import document


class CustomPRNG(abc.ABC): ...


_CUSTOM_PRNG_T = TypeVar("_CUSTOM_PRNG_T", bound=CustomPRNG)
_PRNG_T = Union[np.random.Generator, np.random.RandomState, _CUSTOM_PRNG_T]
_SEED_T = Union[int, None]
PRNG_OR_SEED_LIKE = Union[None, int, np.random.RandomState, np.random.Generator, _CUSTOM_PRNG_T]

document(
    PRNG_OR_SEED_LIKE,
    """A pseudorandom number generator or object that can be converted to one.

    If an integer or None, turns into a `np.random.Generator` seeded with that
    value.

    If none of the above, it is used unmodified. In this case, it is assumed
    that the object implements whatever methods are required for the use case
    at hand. For example, it might be an existing instance of `np.random.Generator`
    or `np.random.RandomState` or a custom pseudorandom number generator implementation
    and in that case, it has to inherit `cirq.value.CustomPRNG`.
    """,
)


@overload
def parse_prng(prng_or_seed: _SEED_T) -> np.random.Generator: ...


@overload
def parse_prng(prng_or_seed: np.random.Generator) -> np.random.Generator: ...


@overload
def parse_prng(prng_or_seed: np.random.RandomState) -> np.random.RandomState: ...


@overload
def parse_prng(prng_or_seed: _CUSTOM_PRNG_T) -> _CUSTOM_PRNG_T: ...


def parse_prng(
    prng_or_seed: PRNG_OR_SEED_LIKE,
) -> Union[np.random.Generator, np.random.RandomState, _CUSTOM_PRNG_T]:
    """Interpret an object as a pseudorandom number generator.

    If `prng_or_seed` is None or an integer, returns `np.random.default_rng(prng_or_seed)`.
    Otherwise, returns `prng_or_seed` unmodified.

    Args:
        prng_or_seed: The object to be used as or converted to a pseudorandom
            number generator.

    Returns:
        The pseudorandom number generator object.
    """
    if prng_or_seed is None or isinstance(prng_or_seed, int):
        return np.random.default_rng(prng_or_seed)
    return prng_or_seed
