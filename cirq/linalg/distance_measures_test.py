# Copyright 2020 The Cirq Developers
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
"""Tests for distance_measures."""

import numpy as np
import pytest

import cirq


def test_fidelity():

    # Known values
    vec1 = np.array([1, 1j, -1, -1j]) * 0.5
    vec2 = np.array([1, -1, 1, -1]) * 0.5
    mat1 = np.outer(vec1, vec1.conj())
    mat2 = np.outer(vec2, vec2.conj())
    mat3 = 0.3 * mat1 + 0.7 * mat2

    np.testing.assert_allclose(cirq.fidelity(vec1, vec1), 1)
    np.testing.assert_allclose(cirq.fidelity(vec2, vec2), 1)
    np.testing.assert_allclose(cirq.fidelity(mat1, mat1), 1)
    np.testing.assert_allclose(cirq.fidelity(mat2, mat2), 1)
    np.testing.assert_allclose(cirq.fidelity(vec1, mat1), 1)
    np.testing.assert_allclose(cirq.fidelity(mat2, vec2), 1)
    np.testing.assert_allclose(cirq.fidelity(vec1, vec2), 0)
    np.testing.assert_allclose(cirq.fidelity(vec1, mat2), 0)
    np.testing.assert_allclose(cirq.fidelity(mat1, vec2), 0)
    np.testing.assert_allclose(cirq.fidelity(vec1, mat3), 0.3)
    np.testing.assert_allclose(cirq.fidelity(mat3, vec2), 0.7)

    # Generic properties
    N = 15
    vec1 = cirq.testing.random_superposition(N)
    vec2 = cirq.testing.random_superposition(N)
    mat1 = cirq.testing.random_density_matrix(N)
    mat2 = cirq.testing.random_density_matrix(N)
    u = cirq.testing.random_unitary(N)

    # Fidelity is symmetric
    np.testing.assert_allclose(cirq.fidelity(vec1, vec2),
                               cirq.fidelity(vec2, vec1))
    np.testing.assert_allclose(cirq.fidelity(vec1, mat1),
                               cirq.fidelity(mat1, vec1))
    np.testing.assert_allclose(cirq.fidelity(mat1, mat2),
                               cirq.fidelity(mat2, mat1))
    # Fidelity is between 0 and 1
    assert 0 <= cirq.fidelity(vec1, vec2) <= 1
    assert 0 <= cirq.fidelity(vec1, mat1) <= 1
    assert 0 <= cirq.fidelity(mat1, mat2) <= 1
    # Fidelity is invariant under unitary transformation
    np.testing.assert_allclose(
        cirq.fidelity(mat1, mat2),
        cirq.fidelity(u @ mat1 @ u.T.conj(), u @ mat2 @ u.T.conj()))

    # Special case of commuting matrices
    d1 = np.random.uniform(size=N)
    d1 /= np.sum(d1)
    d2 = np.random.uniform(size=N)
    d2 /= np.sum(d2)
    mat1 = u @ np.diag(d1) @ u.T.conj()
    mat2 = u @ np.diag(d2) @ u.T.conj()

    np.testing.assert_allclose(cirq.fidelity(mat1, mat2),
                               np.sum(np.sqrt(d1 * d2))**2)

    # Bad shape
    with pytest.raises(ValueError, match='dimensional'):
        _ = cirq.fidelity(np.array([[[1.0]]]), np.array([[[1.0]]]))
