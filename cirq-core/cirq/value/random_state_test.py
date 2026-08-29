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

import numpy as np

import cirq
from cirq.value import random_state


def test_parse_random_state() -> None:
    global_state = np.random.get_state()

    def rand(prng):
        np.random.set_state(global_state)
        return random_state.get_random_array(prng)

    prngs = [
        np.random,
        cirq.value.parse_random_state(np.random),
        cirq.value.parse_random_state(None),
    ]

    vals = [rand(prng) for prng in prngs]
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(*vals)

    seed = np.random.randint(2**31)
    prngs1 = [
        np.random.RandomState(seed),
        cirq.value.parse_random_state(np.random.RandomState(seed)),
        cirq.value.parse_random_state(seed),
    ]
    vals = [random_state.get_random_array(prng) for prng in prngs1]
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(*vals)


def test_get_random_helpers_with_generator() -> None:
    """The helpers should use the `np.random.Generator` spelling of each draw."""

    prng = np.random.default_rng(0)

    assert random_state.get_random_array(prng, (2, 3)).shape == (2, 3)
    assert isinstance(random_state.get_random_array(prng), float)
    assert random_state.get_random_normal_array(prng, (2, 3)).shape == (2, 3)
    assert isinstance(random_state.get_random_normal_array(prng), float)
    assert random_state.get_random_int(prng, 0, 10, size=(2,)).shape == (2,)
    assert 0 <= random_state.get_random_int(prng, 10) < 10


def test_get_random_helpers_with_random_state() -> None:
    """The helpers should use the `np.random.RandomState` spelling of each draw."""

    prng = np.random.RandomState(0)

    assert random_state.get_random_array(prng, (2, 3)).shape == (2, 3)
    assert isinstance(random_state.get_random_array(prng), float)
    assert random_state.get_random_normal_array(prng, (2, 3)).shape == (2, 3)
    assert isinstance(random_state.get_random_normal_array(prng), float)
    assert random_state.get_random_int(prng, 0, 10, size=(2,)).shape == (2,)
    assert 0 <= random_state.get_random_int(prng, 10) < 10
