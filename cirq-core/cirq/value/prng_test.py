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

from typing import List, Union

import pytest
import numpy as np

import cirq


def _sample(prng):
    return tuple(prng.random(10))


def test_parse_rng() -> None:
    eq = cirq.testing.EqualsTester()

    # An `np.random.Generator` or a seed.
    group_inputs: List[Union[int, np.random.Generator]] = [42, np.random.default_rng(42)]
    group: List[np.random.Generator] = [cirq.value.parse_prng(s) for s in group_inputs]
    eq.add_equality_group(*[_sample(g) for g in group])

    # A None seed.
    prng = cirq.value.parse_prng(None)
    eq.add_equality_group(_sample(prng))

    # RandomState PRNG.
    prng = cirq.value.parse_prng(np.random.RandomState(42))
    eq.add_equality_group(_sample(prng))

    # np.random module
    prng = cirq.value.parse_prng(np.random)
    eq.add_equality_group(_sample(prng))

    with pytest.raises(TypeError):
        _ = cirq.value.parse_prng(1.0)
