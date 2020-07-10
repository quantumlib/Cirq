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
"""Tests for measures."""
import numpy as np
import pytest

import cirq

N = 15
VEC1 = cirq.testing.random_superposition(N)
VEC2 = cirq.testing.random_superposition(N)
MAT1 = cirq.testing.random_density_matrix(N)
MAT2 = cirq.testing.random_density_matrix(N)
U = cirq.testing.random_unitary(N)


def test_fidelity_symmetric():
    np.testing.assert_allclose(cirq.fidelity(VEC1, VEC2),
                               cirq.fidelity(VEC2, VEC1))
    np.testing.assert_allclose(cirq.fidelity(VEC1, MAT1),
                               cirq.fidelity(MAT1, VEC1))
    np.testing.assert_allclose(cirq.fidelity(MAT1, MAT2),
                               cirq.fidelity(MAT2, MAT1))


def test_fidelity_between_zero_and_one():
    assert 0 <= cirq.fidelity(VEC1, VEC2) <= 1
    assert 0 <= cirq.fidelity(VEC1, MAT1) <= 1
    assert 0 <= cirq.fidelity(MAT1, MAT2) <= 1


def test_fidelity_invariant_under_unitary_transformation():
    np.testing.assert_allclose(
        cirq.fidelity(MAT1, MAT2),
        cirq.fidelity(U @ MAT1 @ U.T.conj(), U @ MAT2 @ U.T.conj()))


def test_fidelity_commuting_matrices():
    d1 = np.random.uniform(size=N)
    d1 /= np.sum(d1)
    d2 = np.random.uniform(size=N)
    d2 /= np.sum(d2)
    mat1 = U @ np.diag(d1) @ U.T.conj()
    mat2 = U @ np.diag(d2) @ U.T.conj()

    np.testing.assert_allclose(cirq.fidelity(mat1, mat2),
                               np.sum(np.sqrt(d1 * d2))**2)


def test_fidelity_known_values():
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


def test_fidelity_bad_shape():
    with pytest.raises(ValueError, match='dimensional'):
        _ = cirq.fidelity(np.array([[[1.0]]]), np.array([[[1.0]]]))


def test_von_neumann_entropy():
    # 1x1 matrix
    assert cirq.von_neumann_entropy(np.array([[1]])) == 0
    # An EPR pair state (|00> + |11>)(<00| + <11|)
    assert cirq.von_neumann_entropy(
        np.array([1, 0, 0, 1] * np.array([[1], [0], [0], [1]]))) == 0
    # Maximally mixed state
    # yapf: disable
    assert cirq.von_neumann_entropy(np.array(
        [[0.5, 0],
        [0, 0.5]])) == 1
    # 3x3 state
    assert np.isclose(cirq.von_neumann_entropy(
        np.array(
            [[0.5, 0.5j, 1],
            [-0.5j, 0.5, 0],
            [0.7, 0.4, 0.6]])),
                      1.37,
                      atol=1e-01)
    # 4X4 state
    assert np.isclose(cirq.von_neumann_entropy(
        np.array(
            [[0.5, 0.5j, 1, 3],
            [-0.5j, 0.5, 0, 4],
            [0.7, 0.4, 0.6, 5],
            [6, 7, 8, 9]])),
                      1.12,
                      atol=1e-01)
    # yapf: enable
    # 2x2 random unitary, each column as a ket, each ket as a density matrix,
    # linear combination of the two with coefficients 0.1 and 0.9
    res = cirq.testing.random_unitary(2)
    first_column = res[:, 0]
    first_density_matrix = 0.1 * np.outer(first_column, np.conj(first_column))
    second_column = res[:, 1]
    second_density_matrix = 0.9 * np.outer(second_column,
                                           np.conj(second_column))
    assert np.isclose(cirq.von_neumann_entropy(first_density_matrix +
                                               second_density_matrix),
                      0.4689,
                      atol=1e-04)

    assert np.isclose(cirq.von_neumann_entropy(
        np.diag([0, 0, 0.1, 0, 0.2, 0.3, 0.4, 0])),
                      1.8464,
                      atol=1e-04)
