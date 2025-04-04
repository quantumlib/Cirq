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

import random
from typing import Optional, Tuple

import numpy as np
import pytest

import cirq

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1, -1])
H = np.array([[1, 1], [1, -1]]) * np.sqrt(0.5)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
QFT = np.array([[1, 1, 1, 1], [1, 1j, -1, -1j], [1, -1, 1, -1], [1, -1j, -1, 1j]]) * 0.5


def random_real_diagonal_matrix(n: int, d: Optional[int] = None) -> np.ndarray:
    return np.diag([random.random() if d is None or k < d else 0 for k in range(n)])


def random_symmetric_matrix(n: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            p = random.random() * 2 - 1
            m[i, j] = p
            m[j, i] = p
    return m


def random_block_diagonal_symmetric_matrix(*ns: int) -> np.ndarray:
    return cirq.block_diag(*[random_symmetric_matrix(n) for n in ns])


def random_bi_diagonalizable_pair(
    n: int, d1: Optional[int] = None, d2: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    u = cirq.testing.random_orthogonal(n)
    s = random_real_diagonal_matrix(n, d1)
    z = random_real_diagonal_matrix(n, d2)
    v = cirq.testing.random_orthogonal(n)
    a = cirq.dot(u, s, v)
    b = cirq.dot(u, z, v)
    return a, b


def _get_assert_diagonalized_by_str(m, p, d):
    return (
        f'm.round(3) : {np.round(m, 3)}, p.round(3) : {np.round(p, 3)}, '
        f'np.abs(p.T @ m @ p).round(2): {np.abs(d).round(2)}'
    )


def assert_diagonalized_by(m, p, atol: float = 1e-8):
    d = p.T.dot(m).dot(p)

    assert cirq.is_orthogonal(p) and cirq.is_diagonal(
        d, atol=atol
    ), _get_assert_diagonalized_by_str(m, p, d)


def _get_assert_bidiagonalized_by_str(m, p, q, d):
    return (
        f'm.round(3) : {np.round(m, 3)}, p.round(3) : {np.round(p, 3)}, '
        f'q.round(3): {np.round(q, 3)}, np.abs(p.T @ m @ p).round(2): {np.abs(d).round(2)}'
    )


def assert_bidiagonalized_by(m, p, q, rtol: float = 1e-5, atol: float = 1e-8):
    d = p.dot(m).dot(q)

    assert (
        cirq.is_orthogonal(p) and cirq.is_orthogonal(q) and cirq.is_diagonal(d, atol=atol)
    ), _get_assert_bidiagonalized_by_str(m, p, q, d)


@pytest.mark.parametrize(
    'matrix',
    [
        np.array([[0, 0], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [0, 5]]),
        np.array([[1, 1], [1, 1]]),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, 2], [2, 0]]),
        np.array([[-1, 500], [500, 4]]),
        np.array([[-1, 500], [500, -4]]),
        np.array([[1, 3], [3, 7]]),
    ]
    + [random_symmetric_matrix(2) for _ in range(10)]
    + [random_symmetric_matrix(4) for _ in range(10)]
    + [random_symmetric_matrix(k) for k in range(1, 10)],
)
def test_diagonalize_real_symmetric_matrix(matrix):
    p = cirq.diagonalize_real_symmetric_matrix(matrix)
    assert_diagonalized_by(matrix, p)


@pytest.mark.parametrize(
    'matrix',
    [
        np.array([[0, 1], [0, 0]]),
        np.array([[1, 1], [0, 1]]),
        np.array([[1, 1j], [-1j, 1]]),
        np.array([[1, 1j], [1j, 1]]),
        np.array([[3, 1], [7, 3]]),
    ],
)
def test_diagonalize_real_symmetric_matrix_fails(matrix):
    with pytest.raises(ValueError):
        _ = cirq.diagonalize_real_symmetric_matrix(matrix)


def test_diagonalize_real_symmetric_matrix_assertion_error():
    with pytest.raises(AssertionError):
        matrix = np.array([[0.5, 0], [0, 1]])
        m = np.array([[0, 1], [0, 0]])
        p = cirq.diagonalize_real_symmetric_matrix(matrix)
        assert_diagonalized_by(m, p)


@pytest.mark.parametrize(
    's,m',
    [
        ([1, 1], np.eye(2)),
        ([-1, -1], np.eye(2)),
        ([2, 1], np.eye(2)),
        ([2, 0], np.eye(2)),
        ([0, 0], np.eye(2)),
        ([1, 1], [[0, 1], [1, 0]]),
        ([2, 2], [[0, 1], [1, 0]]),
        ([1, 1], [[1, 3], [3, 6]]),
        ([2, 2, 1], [[1, 3, 0], [3, 6, 0], [0, 0, 1]]),
        ([2, 1, 1], [[-5, 0, 0], [0, 1, 3], [0, 3, 6]]),
    ]
    + [([6, 6, 5, 5, 5], random_block_diagonal_symmetric_matrix(2, 3)) for _ in range(10)],
)
def test_simultaneous_diagonalize_real_symmetric_matrix_vs_singulars(s, m):
    m = np.array(m)
    s = np.diag(s)
    p = cirq.diagonalize_real_symmetric_and_sorted_diagonal_matrices(m, s)
    assert_diagonalized_by(s, p)
    assert_diagonalized_by(m, p)
    assert np.allclose(s, p.T.dot(s).dot(p))


@pytest.mark.parametrize(
    's,m,match',
    [
        ([1, 2], np.eye(2), 'must be real diagonal descending'),
        ([2, 1], [[0, 1], [1, 0]], 'must commute'),
        ([1, 0], [[1, 3], [3, 6]], 'must commute'),
        ([2, 1, 1], [[1, 3, 0], [3, 6, 0], [0, 0, 1]], 'must commute'),
        ([2, 2, 1], [[-5, 0, 0], [0, 1, 3], [0, 3, 6]], 'must commute'),
        ([3, 2, 1], QFT, 'must be real symmetric'),
    ],
)
def test_simultaneous_diagonalize_real_symmetric_matrix_vs_singulars_fail(s, m, match: str):
    m = np.array(m)
    s = np.diag(s)
    with pytest.raises(ValueError, match=match):
        cirq.diagonalize_real_symmetric_and_sorted_diagonal_matrices(m, s)


@pytest.mark.parametrize(
    'a,b',
    [
        (np.zeros((0, 0)), np.zeros((0, 0))),
        (np.eye(2), np.eye(2)),
        (np.eye(4), np.eye(4)),
        (np.eye(4), np.zeros((4, 4))),
        (H, H),
        (cirq.kron(np.eye(2), H), cirq.kron(H, np.eye(2))),
        (cirq.kron(np.eye(2), Z), cirq.kron(X, np.eye(2))),
    ]
    + [random_bi_diagonalizable_pair(2) for _ in range(10)]
    + [random_bi_diagonalizable_pair(4) for _ in range(10)]
    + [
        random_bi_diagonalizable_pair(4, d1, d2)
        for _ in range(10)
        for d1 in range(4)
        for d2 in range(4)
    ]
    + [random_bi_diagonalizable_pair(k) for k in range(1, 10)],
)
def test_bidiagonalize_real_matrix_pair_with_symmetric_products(a, b):
    a = np.array(a)
    b = np.array(b)
    p, q = cirq.bidiagonalize_real_matrix_pair_with_symmetric_products(a, b)
    assert_bidiagonalized_by(a, p, q)
    assert_bidiagonalized_by(b, p, q)


@pytest.mark.parametrize(
    'a,b,match',
    [
        [X, Z, 'must be symmetric'],
        [Y, np.eye(2), 'must be real'],
        [np.eye(2), Y, 'must be real'],
        [np.eye(5), np.eye(4), 'shapes'],
        [e * cirq.testing.random_orthogonal(4) for e in random_bi_diagonalizable_pair(4)]
        + ['must be symmetric'],
        [np.array([[1, 1], [1, 0]]), np.array([[1, 1], [0, 1]]), 'mat1.T @ mat2 must be symmetric'],
        [np.array([[1, 1], [1, 0]]), np.array([[1, 0], [1, 1]]), 'mat1 @ mat2.T must be symmetric'],
    ],
)
def test_bidiagonalize_real_fails(a, b, match: str):
    a = np.array(a)
    b = np.array(b)
    with pytest.raises(ValueError, match=match):
        cirq.bidiagonalize_real_matrix_pair_with_symmetric_products(a, b)


def test_bidiagonalize__assertion_error():
    with pytest.raises(AssertionError):
        a = np.diag([0, 1])
        assert_bidiagonalized_by(a, a, a)


@pytest.mark.parametrize(
    'mat',
    [
        np.diag([1]),
        np.diag([1j]),
        np.diag([-1]),
        np.eye(4),
        # Known near-identity failure for naive QZ decomposition strategy:
        np.array(
            [
                [1.0000000000000002 + 0j, 0, 0, 4.266421588589642e-17j],
                [0, 1.0000000000000002, -4.266421588589642e-17j, 0],
                [0, 4.266421588589642e-17j, 1.0000000000000002, 0],
                [-4.266421588589642e-17j, 0, 0, 1.0000000000000002],
            ]
        ),
        SWAP,
        CNOT,
        Y,
        H,
        cirq.kron(H, H),
        cirq.kron(Y, Y),
        QFT,
    ]
    + [cirq.testing.random_unitary(2) for _ in range(10)]
    + [cirq.testing.random_unitary(4) for _ in range(10)]
    + [cirq.testing.random_unitary(k) for k in range(1, 10)],
)
def test_bidiagonalize_unitary_with_special_orthogonals(mat):
    p, d, q = cirq.bidiagonalize_unitary_with_special_orthogonals(mat)
    assert cirq.is_special_orthogonal(p)
    assert cirq.is_special_orthogonal(q)
    assert np.allclose(p.dot(mat).dot(q), np.diag(d))
    assert_bidiagonalized_by(mat, p, q)


@pytest.mark.parametrize(
    'mat',
    [np.diag([0]), np.diag([0.5]), np.diag([1, 0]), np.diag([0.5, 2]), np.array([[0, 1], [0, 0]])],
)
def test_bidiagonalize_unitary_fails(mat):
    with pytest.raises(ValueError):
        cirq.bidiagonalize_unitary_with_special_orthogonals(mat)
