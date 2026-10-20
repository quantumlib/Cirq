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
import pytest

import cirq


def test_parse_random_state() -> None:
    global_state = np.random.get_state()

    def rand(prng):
        np.random.set_state(global_state)
        return prng.rand()

    prngs = [np.random, cirq.value.parse_random_state(None)]
    vals = [rand(prng) for prng in prngs]
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(*vals)

    seed = np.random.randint(2**31)
    prngs1 = [
        np.random.RandomState(seed),
        cirq.value.parse_random_state(np.random.RandomState(seed)),
        cirq.value.parse_random_state(seed),
    ]
    vals = [prng.rand() for prng in prngs1]
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(*vals)


def test_parse_random_state_generator() -> None:
    generator = np.random.default_rng(42)
    prng = cirq.value.parse_random_state(generator)
    assert isinstance(prng, np.random.RandomState)
    # The RandomState shares the generator's bit generator, so draws interleave on one stream.
    expected = np.random.default_rng(42)
    assert prng.random_sample() == expected.random()
    assert generator.random() == expected.random()


def test_parse_random_generator() -> None:
    prngs = [cirq.value.parse_random_generator(42), cirq.value.parse_random_generator(np.int32(42))]
    vals = [prng.random() for prng in prngs]
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(*vals)

    prng = cirq.value.parse_random_generator(None)
    assert isinstance(prng, np.random.Generator)

    prngs1 = [
        cirq.value.parse_random_generator(np.random.RandomState(42)),
        cirq.value.parse_random_generator(np.random.RandomState(42)),
    ]
    vals = [prng.random() for prng in prngs1] + [np.random.RandomState(42).rand()]
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(*vals)

    generator = np.random.default_rng(42)
    assert cirq.value.parse_random_generator(generator) is generator


def test_parse_random_generator_default_rng() -> None:
    eq = cirq.testing.EqualsTester()
    prng = cirq.value.parse_random_generator(None)

    eq.add_equality_group(prng.random())
    eq.add_equality_group(cirq.value.parse_random_generator(None).random())

    cirq.value.reseed_default_prng(1234)
    val0 = prng.random()
    cirq.value.reseed_default_prng(1234)
    val1 = prng.random()
    eq.add_equality_group(val0, val1)

    cirq.value.reseed_default_prng(1234)
    val1_0 = prng.spawn(1)[0].random()
    cirq.value.reseed_default_prng(1234)
    val1_1 = prng.spawn(1)[0].random()
    eq.add_equality_group(val1_0, val1_1)

    assert prng is cirq.value.parse_random_generator(None)


def test_parse_random_generator_invalid() -> None:
    invalid_inputs = [3.14, 42.0, np.float64(42), "42", np.random]

    for invalid_input in invalid_inputs:
        with pytest.raises(TypeError):
            cirq.value.parse_random_generator(invalid_input)  # type: ignore[arg-type]
