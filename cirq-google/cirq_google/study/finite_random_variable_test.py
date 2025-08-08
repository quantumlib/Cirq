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


import random

import pytest
import sympy

import cirq
import cirq_google

KEY = sympy.Symbol('t')
DIST = {0.1: 1, 0.5: 2, 0.9: 1}
SEED = 42
LENGTH = 100


def test_init_and_properties():
    """Tests if the constructor correctly initializes all attributes."""
    sweep = cirq_google.study.FiniteRandomVariable(
        key=KEY, distribution=DIST, seed=SEED, length=LENGTH, metadata="test_meta"
    )
    assert sweep.key == str(KEY)
    assert sweep.keys == [str(KEY)]
    assert sweep.distribution == DIST
    assert sweep.seed == SEED
    assert sweep.length == LENGTH
    assert sweep.metadata == "test_meta"


def test_init_validation():
    """Tests that the constructor raises errors for invalid inputs."""
    with pytest.raises(ValueError, match="distribution must be a non-empty dictionary"):
        cirq_google.study.FiniteRandomVariable(key=KEY, distribution={}, seed=SEED, length=LENGTH)
    with pytest.raises(ValueError, match="length must be a positive integer"):
        cirq_google.study.FiniteRandomVariable(key=KEY, distribution=DIST, seed=SEED, length=0)
    with pytest.raises(ValueError, match="distribution keys must be numbers"):
        cirq_google.study.FiniteRandomVariable(
            key=KEY, distribution={'a': 1}, seed=SEED, length=LENGTH
        )
    with pytest.raises(ValueError, match="distribution values.*must be non-negative"):
        cirq_google.study.FiniteRandomVariable(
            key=KEY, distribution={0.1: -5}, seed=SEED, length=LENGTH
        )


def test_len():
    """Tests that len(sweep) returns the correct length."""
    sweep = cirq_google.study.FiniteRandomVariable(
        key=KEY, distribution=DIST, seed=SEED, length=LENGTH
    )
    assert len(sweep) == LENGTH


def test_reproducibility_and_values():
    """Tests that two identical sweeps produce the same sequence of values."""
    sweep1 = cirq_google.study.FiniteRandomVariable(
        key=KEY, distribution=DIST, seed=SEED, length=LENGTH
    )
    sweep2 = cirq_google.study.FiniteRandomVariable(
        key=KEY, distribution=DIST, seed=SEED, length=LENGTH
    )

    vals1 = list(sweep1)
    vals2 = list(sweep2)

    assert len(vals1) == LENGTH
    # Verify that the sequence is deterministic and reproducible
    assert vals1 == vals2


def test_iteration_caching():
    """Tests that values are generated once and then cached."""
    sweep = cirq_google.study.FiniteRandomVariable(
        key=KEY, distribution=DIST, seed=SEED, length=LENGTH
    )
    sweep2 = cirq_google.study.FiniteRandomVariable(
        key=KEY, distribution=DIST, seed=SEED, length=LENGTH
    )

    vals1 = list(sweep)
    vals2 = list(sweep2)
    assert vals1 == vals2


def test_rng_state_is_restored():
    """Ensures the sweep does not affect the global random number generator state."""
    global_seed = 123
    random.seed(global_seed)

    # Create and iterate over a sweep with a different, independent seed
    sweep = cirq_google.study.FiniteRandomVariable(key=KEY, distribution=DIST, seed=999, length=20)
    _ = list(sweep)  # This should use its own RNG state and restore the global one

    # Generate another value from the global RNG
    value_after_sweep = random.random()

    # Now, check if the global RNG state was truly unaffected
    random.seed(global_seed)
    expected_next_value = random.random()

    assert value_after_sweep == expected_next_value


def test_equality():
    """Tests the __eq__ method for comparing sweep instances."""
    sweep1 = cirq_google.study.FiniteRandomVariable(
        key=KEY, distribution=DIST, seed=SEED, length=LENGTH
    )
    sweep2 = cirq_google.study.FiniteRandomVariable(
        key=KEY, distribution=DIST, seed=SEED, length=LENGTH
    )
    sweep_diff_seed = cirq_google.study.FiniteRandomVariable(
        key=KEY, distribution=DIST, seed=SEED + 1, length=LENGTH
    )
    sweep_diff_len = cirq_google.study.FiniteRandomVariable(
        key=KEY, distribution=DIST, seed=SEED, length=LENGTH + 1
    )

    assert sweep1 == sweep2
    assert sweep1 != sweep_diff_seed
    assert sweep1 != sweep_diff_len
    assert sweep1 != "not a sweep"


def test_repr():
    """Tests that the repr is valid and can be evaluated to recreate the object."""
    sweep = cirq_google.study.FiniteRandomVariable(
        key=KEY, distribution=DIST, seed=SEED, length=LENGTH, metadata={'data': 1}
    )
    print(repr(sweep))
    # cirq.testing.assert_equivalent_repr evaluates the repr and checks equality
    cirq.testing.assert_equivalent_repr(sweep, setup_code='import sympy\nimport cirq_google')


def test_json_serialization():
    """Tests that the sweep can be correctly serialized and deserialized via JSON.

    This class is still experimental, so we will not add it to the main JSON resolvers yet.
    """
    sweep = cirq_google.study.FiniteRandomVariable(
        key=KEY, distribution=DIST, seed=SEED, length=LENGTH, metadata={'data': [1, 2]}
    )
    sweep_reconstruct = cirq.read_json(
        json_text=cirq.to_json(sweep),
        resolvers=[
            lambda s: (
                cirq_google.study.FiniteRandomVariable if s == "FiniteRandomVariable" else None
            )
        ],
    )
    assert sweep_reconstruct == sweep
