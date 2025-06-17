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

import pytest
import numpy as np

import cirq


def test_parse_prng_generator_passthrough():
    """Test that passing an existing Generator returns the same object."""
    rng = np.random.default_rng(12345)
    assert cirq.value.parse_prng(rng) is rng


def test_parse_prng_none():
    """Test that passing None returns a new Generator instance."""
    rng1 = cirq.value.parse_prng(None)
    rng2 = cirq.value.parse_prng(None)
    assert rng1 is not rng2
    assert type(rng1) is np.random.Generator
    assert type(rng2) is np.random.Generator


def test_parse_prng_int_seeding():
    """Test that integer seeds create predictable Generators."""
    rng_int = cirq.value.parse_prng(42)
    rng_npint = cirq.value.parse_prng(np.int64(42))
    assert rng_int.random() == rng_npint.random()

    rng_different_seed = cirq.value.parse_prng(43)
    rng_int = cirq.value.parse_prng(42)
    assert rng_int.random() != rng_different_seed.random()


def test_parse_prng_module_disallowed():
    """Test that passing the np.random module raises TypeError."""
    with pytest.raises(TypeError, match="not supported"):
        cirq.value.parse_prng(np.random)


def test_parse_prng_invalid_types():
    """Test that unsupported types raise TypeError."""

    match = "cannot be converted"
    with pytest.raises(TypeError, match=match):
        cirq.value.parse_prng(1.0)

    with pytest.raises(TypeError, match=match):
        cirq.value.parse_prng("not a seed")

    with pytest.raises(TypeError, match=match):
        cirq.value.parse_prng([1, 2, 3])

    with pytest.raises(TypeError, match=match):
        cirq.value.parse_prng(object())


def test_parse_prng_equality_tester_on_output():
    """Use EqualsTester to verify output consistency for valid inputs."""
    eq = cirq.testing.EqualsTester()

    eq.add_equality_group(
        cirq.value.parse_prng(42).random(),
        cirq.value.parse_prng(np.int32(42)).random(),
        cirq.value.parse_prng(np.random.default_rng(42)).random(),
    )

    eq.add_equality_group(
        cirq.value.parse_prng(np.random.RandomState(50)).random(),
        cirq.value.parse_prng(np.random.RandomState(50)).random(),
    )

    eq.add_equality_group(cirq.value.parse_prng(None).random())
