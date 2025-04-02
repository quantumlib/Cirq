# Copyright 2018 The Cirq Developers
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

import numpy as np
import pytest

from cirq.linalg import is_orthogonal, is_special_orthogonal, is_special_unitary, is_unitary
from cirq.testing import (
    assert_allclose_up_to_global_phase,
    random_density_matrix,
    random_orthogonal,
    random_special_orthogonal,
    random_special_unitary,
    random_superposition,
    random_unitary,
)


@pytest.mark.parametrize('dim', range(1, 10))
def test_random_superposition(dim):
    state = random_superposition(dim)

    assert dim == len(state)
    assert np.isclose(np.linalg.norm(state), 1.0)


def test_random_superposition_deterministic_given_seed():
    state1 = random_superposition(10, random_state=1234)
    state2 = random_superposition(10, random_state=1234)

    np.testing.assert_equal(state1, state2)


@pytest.mark.parametrize('dim', range(1, 10))
def test_random_density_matrix(dim):
    state = random_density_matrix(dim)

    assert state.shape == (dim, dim)
    np.testing.assert_allclose(np.trace(state), 1)
    np.testing.assert_allclose(state, state.T.conj())
    eigs, _ = np.linalg.eigh(state)
    assert np.all(eigs >= 0)


def test_random_density_matrix_deterministic_given_seed():
    state1 = random_density_matrix(10, random_state=1234)
    state2 = random_density_matrix(10, random_state=1234)

    np.testing.assert_equal(state1, state2)


def test_random_unitary():
    u1 = random_unitary(2)
    u2 = random_unitary(2)
    assert is_unitary(u1)
    assert is_unitary(u2)
    assert not np.allclose(u1, u2)


def test_random_orthogonal():
    o1 = random_orthogonal(2)
    o2 = random_orthogonal(2)
    assert is_orthogonal(o1)
    assert is_orthogonal(o2)
    assert not np.allclose(o1, o2)


def test_random_orthogonal_deterministic_given_seed():
    o1 = random_orthogonal(2, random_state=1234)
    o2 = random_orthogonal(2, random_state=1234)

    np.testing.assert_equal(o1, o2)


def test_random_special_unitary():
    u1 = random_special_unitary(2)
    u2 = random_special_unitary(2)
    assert is_special_unitary(u1)
    assert is_special_unitary(u2)
    assert not np.allclose(u1, u2)


def test_seeded_special_unitary():
    u1 = random_special_unitary(2, random_state=np.random.RandomState(1))
    u2 = random_special_unitary(2, random_state=np.random.RandomState(1))
    u3 = random_special_unitary(2, random_state=np.random.RandomState(2))
    assert np.allclose(u1, u2)
    assert not np.allclose(u1, u3)


def test_random_special_orthogonal():
    o1 = random_special_orthogonal(2)
    o2 = random_special_orthogonal(2)
    assert is_special_orthogonal(o1)
    assert is_special_orthogonal(o2)
    assert not np.allclose(o1, o2)


def test_random_special_orthogonal_deterministic_given_seed():
    o1 = random_special_orthogonal(2, random_state=1234)
    o2 = random_special_orthogonal(2, random_state=1234)

    np.testing.assert_equal(o1, o2)


def test_assert_allclose_up_to_global_phase():
    assert_allclose_up_to_global_phase(np.array([[1]]), np.array([[1j]]), atol=0)

    with pytest.raises(AssertionError):
        assert_allclose_up_to_global_phase(np.array([[1]]), np.array([[2]]), atol=0)

    assert_allclose_up_to_global_phase(
        np.array([[1e-8, -1, 1e-8]]), np.array([[1e-8, 1, 1e-8]]), atol=1e-6
    )

    with pytest.raises(AssertionError):
        assert_allclose_up_to_global_phase(
            np.array([[1e-4, -1, 1e-4]]), np.array([[1e-4, 1, 1e-4]]), atol=1e-6
        )

    assert_allclose_up_to_global_phase(
        np.array([[1, 2], [3, 4]]), np.array([[-1, -2], [-3, -4]]), atol=0
    )
