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

import numpy as np
import pytest

import cirq


def assert_valid_density_matrix(matrix, num_qubits=1):
    np.testing.assert_almost_equal(
        cirq.to_valid_density_matrix(matrix, num_qubits=num_qubits,
                                     dtype=matrix.dtype), matrix)


def test_to_valid_density_matrix_from_density_matrix():
    assert_valid_density_matrix(np.array([[1, 0], [0, 0]]))
    assert_valid_density_matrix(np.array([[0.5, 0], [0, 0.5]]))
    assert_valid_density_matrix(np.array([[0.5, 0.5], [0.5, 0.5]]))
    assert_valid_density_matrix(np.array([[0.5, 0.2], [0.2, 0.5]]))
    assert_valid_density_matrix(np.array([[0.5, 0.5j], [-0.5j, 0.5]]))
    assert_valid_density_matrix(
        np.array([[0.5, 0.2 - 0.2j], [0.2 + 0.2j, 0.5]]))
    assert_valid_density_matrix(np.eye(4) / 4.0, num_qubits=2)
    assert_valid_density_matrix(np.diag([1, 0, 0, 0]), num_qubits=2)
    assert_valid_density_matrix(np.ones([4, 4]) / 4.0, num_qubits=2)
    assert_valid_density_matrix(np.diag([0.2, 0.8, 0, 0]), num_qubits=2)
    assert_valid_density_matrix(np.array(
        [[0.2, 0, 0, 0.2 - 0.3j],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0.2 + 0.3j, 0, 0, 0.8]]),
        num_qubits=2)


def test_to_valid_density_matrix_not_square():
    with pytest.raises(ValueError, match='square'):
        cirq.to_valid_density_matrix(np.array([[1, 0]]), num_qubits=1)
    with pytest.raises(ValueError, match='square'):
        cirq.to_valid_density_matrix(np.array([[1], [0]]), num_qubits=1)


def test_to_valid_density_matrix_size_mismatch_num_qubits():
    with pytest.raises(ValueError, match='size'):
        cirq.to_valid_density_matrix(np.array([[1, 0], [0, 0]]), num_qubits=2)
    with pytest.raises(ValueError, match='size'):
        cirq.to_valid_density_matrix(np.eye(4) / 4.0, num_qubits=1)


def test_to_valid_density_matrix_not_hermitian():
    with pytest.raises(ValueError, match='hermitian'):
        cirq.to_valid_density_matrix(np.array([[1, 0.1], [0, 0]]), num_qubits=1)
    with pytest.raises(ValueError, match='hermitian'):
        cirq.to_valid_density_matrix(np.array([[0.5, 0.5j], [0.5, 0.5j]]),
                                     num_qubits=1)
    with pytest.raises(ValueError, match='hermitian'):
        cirq.to_valid_density_matrix(
            np.array(
                [[0.2, 0, 0, -0.2 - 0.3j],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0.2 + 0.3j, 0, 0, 0.8]]),
            num_qubits=2)


def test_to_valid_density_matrix_not_unit_trace():
    with pytest.raises(ValueError, match='trace 1'):
        cirq.to_valid_density_matrix(np.array([[1, 0], [0, 0.1]]), num_qubits=1)
    with pytest.raises(ValueError, match='trace 1'):
        cirq.to_valid_density_matrix(np.array([[1, 0], [0, -0.1]]),
                                     num_qubits=1)
    with pytest.raises(ValueError, match='trace 1'):
        cirq.to_valid_density_matrix(np.zeros([2, 2]), num_qubits=1)


def test_to_valid_density_matrix_not_positive_semidefinite():
    with pytest.raises(ValueError, match='positive semidefinite'):
        cirq.to_valid_density_matrix(
            np.array([[1.1, 0], [0, -0.1]], dtype=np.complex64), num_qubits=1)
    with pytest.raises(ValueError, match='positive semidefinite'):
        cirq.to_valid_density_matrix(
            np.array([[0.6, 0.5], [0.5, 0.4]], dtype=np.complex64),
            num_qubits=1)


def test_to_valid_density_matrix_wrong_dtype():
    with pytest.raises(ValueError, match='dtype'):
        cirq.to_valid_density_matrix(
            np.array([[1, 0], [0, 0]], dtype=np.complex64),
            num_qubits=1, dtype=np.complex128)


def test_to_valid_density_matrix_from_state():
    np.testing.assert_almost_equal(
        cirq.to_valid_density_matrix(
            density_matrix_rep=np.array([1, 0], dtype=np.complex64),
            num_qubits=1),
        np.array([[1, 0], [0, 0]]))
    np.testing.assert_almost_equal(
        cirq.to_valid_density_matrix(
            density_matrix_rep=np.array([np.sqrt(0.3), np.sqrt(0.7)],
                                        dtype=np.complex64),
            num_qubits=1),
        np.array([[0.3, np.sqrt(0.3 * 0.7)], [np.sqrt(0.3 * 0.7), 0.7]]))
    np.testing.assert_almost_equal(
        cirq.to_valid_density_matrix(
            density_matrix_rep=np.array([np.sqrt(0.5), np.sqrt(0.5) * 1j],
                                        dtype=np.complex64),
            num_qubits=1),
        np.array([[0.5, -0.5j], [0.5j, 0.5]]))
    np.testing.assert_almost_equal(
        cirq.to_valid_density_matrix(
            density_matrix_rep=np.array([0.5] * 4, dtype=np.complex64),
            num_qubits=2),
        0.25 * np.ones((4, 4)))


def test_to_valid_density_matrix_from_state_invalid_state():
    with pytest.raises(ValueError, match="2 qubits"):
        cirq.to_valid_density_matrix(np.array([1, 0]), num_qubits=2)


def test_to_valid_density_matrix_from_computational_basis():
    np.testing.assert_almost_equal(
        cirq.to_valid_density_matrix(density_matrix_rep=0, num_qubits=1),
        np.array([[1, 0], [0, 0]]))
    np.testing.assert_almost_equal(
        cirq.to_valid_density_matrix(density_matrix_rep=1, num_qubits=1),
        np.array([[0, 0], [0, 1]]))
    np.testing.assert_almost_equal(
        cirq.to_valid_density_matrix(density_matrix_rep=2, num_qubits=2),
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]))


def test_to_valid_density_matrix_from_state_invalid_computational_basis():
    with pytest.raises(ValueError, match="positive"):
        cirq.to_valid_density_matrix(-1, num_qubits=2)
