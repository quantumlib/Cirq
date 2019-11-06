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
import pytest
import numpy as np

import cirq


def test_dot():
    assert cirq.dot(2) == 2
    assert cirq.dot(2.5, 2.5) == 6.25

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    assert cirq.dot(a) is not a
    np.testing.assert_allclose(cirq.dot(a),
                               a,
                               atol=1e-8)
    np.testing.assert_allclose(cirq.dot(a, b),
                               np.dot(a, b),
                               atol=1e-8)
    np.testing.assert_allclose(cirq.dot(a, b, a),
                               np.dot(np.dot(a, b), a),
                               atol=1e-8)

    # Invalid use
    with pytest.raises(ValueError):
        cirq.dot()


def test_kron_multiplies_sizes():
    assert cirq.kron(np.array([1, 2])).shape == (1, 2)
    assert cirq.kron(np.array([1, 2]), shape_len=1).shape == (2,)
    assert cirq.kron(np.array([1, 2]), np.array([3, 4, 5]),
                     shape_len=1).shape == (6,)
    assert cirq.kron(shape_len=0).shape == ()
    assert cirq.kron(shape_len=1).shape == (1,)
    assert cirq.kron(shape_len=2).shape == (1, 1)

    assert np.allclose(cirq.kron(1j, np.array([2, 3])), np.array([2j, 3j]))
    assert np.allclose(cirq.kron(), np.eye(1))
    assert np.allclose(cirq.kron(np.eye(1)), np.eye(1))
    assert np.allclose(cirq.kron(np.eye(2)), np.eye(2))
    assert np.allclose(cirq.kron(np.eye(1), np.eye(1)), np.eye(1))
    assert np.allclose(cirq.kron(np.eye(1), np.eye(2)), np.eye(2))
    assert np.allclose(cirq.kron(np.eye(2), np.eye(3)), np.eye(6))
    assert np.allclose(cirq.kron(np.eye(2), np.eye(3), np.eye(4)),
                       np.eye(24))


def test_kron_spreads_values():
    u = np.array([[2, 3], [5, 7]])

    assert np.allclose(
        cirq.kron(np.eye(2), u),
        np.array([[2, 3, 0, 0], [5, 7, 0, 0], [0, 0, 2, 3], [0, 0, 5, 7]]))

    assert np.allclose(
        cirq.kron(u, np.eye(2)),
        np.array([[2, 0, 3, 0], [0, 2, 0, 3], [5, 0, 7, 0], [0, 5, 0, 7]]))

    assert np.allclose(
        cirq.kron(u, u),
        np.array([[4, 6, 6, 9], [10, 14, 15, 21], [10, 15, 14, 21],
                [25, 35, 35, 49]]))


def test_acts_like_kron_multiplies_sizes():
    assert np.allclose(cirq.kron_with_controls(), np.eye(1))
    assert np.allclose(
        cirq.kron_with_controls(np.eye(2), np.eye(3), np.eye(4)),
        np.eye(24))

    u = np.array([[2, 3], [5, 7]])
    assert np.allclose(
        cirq.kron_with_controls(u, u),
        np.array([[4, 6, 6, 9], [10, 14, 15, 21], [10, 15, 14, 21],
                [25, 35, 35, 49]]))


def test_supports_controls():
    u = np.array([[2, 3], [5, 7]])
    assert np.allclose(
        cirq.kron_with_controls(cirq.CONTROL_TAG),
        np.array([[1, 0], [0, 1]]))
    assert np.allclose(
        cirq.kron_with_controls(cirq.CONTROL_TAG, u),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 3], [0, 0, 5, 7]]))
    assert np.allclose(
        cirq.kron_with_controls(u, cirq.CONTROL_TAG),
        np.array([[1, 0, 0, 0], [0, 2, 0, 3], [0, 0, 1, 0], [0, 5, 0, 7]]))


def test_block_diag():
    assert np.allclose(
        cirq.block_diag(),
        np.zeros((0, 0)))

    assert np.allclose(
        cirq.block_diag(
            np.array([[1, 2],
                    [3, 4]])),
        np.array([[1, 2],
                [3, 4]]))

    assert np.allclose(
        cirq.block_diag(
            np.array([[1, 2],
                    [3, 4]]),
            np.array([[4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12]])),
        np.array([[1, 2, 0, 0, 0],
                [3, 4, 0, 0, 0],
                [0, 0, 4, 5, 6],
                [0, 0, 7, 8, 9],
                [0, 0, 10, 11, 12]]))


def test_block_diag_dtype():
    assert cirq.block_diag().dtype == np.complex128

    assert (cirq.block_diag(np.array([[1]], dtype=np.int8)).dtype ==
            np.int8)

    assert cirq.block_diag(
            np.array([[1]], dtype=np.float32),
            np.array([[2]], dtype=np.float32)).dtype == np.float32

    assert cirq.block_diag(
            np.array([[1]], dtype=np.float64),
            np.array([[2]], dtype=np.float64)).dtype == np.float64

    assert cirq.block_diag(
            np.array([[1]], dtype=np.float32),
            np.array([[2]], dtype=np.float64)).dtype == np.float64

    assert cirq.block_diag(
            np.array([[1]], dtype=np.float32),
            np.array([[2]], dtype=np.complex64)).dtype == np.complex64

    assert cirq.block_diag(
            np.array([[1]], dtype=np.int),
            np.array([[2]], dtype=np.complex128)).dtype == np.complex128
