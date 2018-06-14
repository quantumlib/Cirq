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
from typing import Tuple, Optional

import numpy as np
import pytest

from cirq.linalg import combinators
from cirq.linalg import diagonalize
from cirq.linalg import predicates
from cirq.linalg.tolerance import Tolerance
from cirq import testing


X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1, -1])
H = np.array([[1, 1], [1, -1]]) * np.sqrt(0.5)
SWAP = np.array([[1, 0, 0, 0],
               [0, 0, 1, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 1]])
CNOT = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 1],
               [0, 0, 1, 0]])
QFT = np.array([[1, 1, 1, 1],
              [1, 1j, -1, -1j],
              [1, -1, 1, -1],
              [1, -1j, -1, 1j]]) * 0.5


def random_real_diagonal_matrix(n: int, d: Optional[int] = None) -> np.ndarray:
    return np.diag([random.random() if d is None or k < d else 0
                    for k in range(n)])


def random_symmetric_matrix(n: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            p = random.random() * 2 - 1
            m[i, j] = p
            m[j, i] = p
    return m


def random_block_diagonal_symmetric_matrix(*ns: int) -> np.ndarray:
    return combinators.block_diag(*[random_symmetric_matrix(n) for n in ns])


def random_bi_diagonalizable_pair(
        n: int, d1: Optional[int] = None, d2: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    u = testing.random_orthogonal(n)
    s = random_real_diagonal_matrix(n, d1)
    z = random_real_diagonal_matrix(n, d2)
    v = testing.random_orthogonal(n)
    a = combinators.dot(u, s, v)
    b = combinators.dot(u, z, v)
    return a, b


def assertdiagonalized_by(m, p, tol=Tolerance.DEFAULT):
    d = p.T.dot(m).dot(p)

    try:
        assert predicates.is_orthogonal(p)
        assert predicates.is_diagonal(d, tol)
    except AssertionError:
        # coverage: ignore

        print("m.round(3)")
        print(np.round(m, 3))

        print("p.round(3)")
        print(np.round(p, 3))

        print("np.log10(np.abs(p.T @ m @ p)).round(2)")
        print(np.log10(np.abs(d)).round(2))

        raise


def assert_bidiagonalized_by(m, p, q, tol=Tolerance.DEFAULT):
    d = p.dot(m).dot(q)

    try:
        assert predicates.is_orthogonal(p)
        assert predicates.is_orthogonal(q)
        assert predicates.is_diagonal(d, tol)
    except AssertionError:
        # coverage: ignore

        print("m.round(3)")
        print(np.round(m, 3))

        print("p.round(3)")
        print(np.round(p, 3))

        print("q.round(3)")
        print(np.round(q, 3))

        print("np.log10(np.abs(p @ m @ q)).round(2)")
        print(np.log10(np.abs(d)).round(2))

        raise


@pytest.mark.parametrize('matrix', [
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
] + [
    random_symmetric_matrix(2) for _ in
    range(10)
] + [
    random_symmetric_matrix(4) for _ in
    range(10)
] + [
    random_symmetric_matrix(k) for k in
    range(1, 10)
])
def test_diagonalize_real_symmetric_matrix(matrix):
    p = diagonalize.diagonalize_real_symmetric_matrix(matrix)
    assertdiagonalized_by(matrix, p)


@pytest.mark.parametrize('matrix', [
    np.array([[0, 1], [0, 0]]),
    np.array([[1, 1], [0, 1]]),
    np.array([[1, 1j], [-1j, 1]]),
    np.array([[1, 1j], [1j, 1]]),
    np.array([[3, 1], [7, 3]]),
])
def test_diagonalize_real_symmetric_matrix_fails(matrix):
    with pytest.raises(ValueError):
        _ = diagonalize.diagonalize_real_symmetric_matrix(matrix)


@pytest.mark.parametrize('s,m', [
    ([1, 1], np.eye(2)),
    ([-1, -1], np.eye(2)),
    ([2, 1], np.eye(2)),
    ([2, 0], np.eye(2)),
    ([0, 0], np.eye(2)),
    ([1, 1], [[0, 1], [1, 0]]),
    ([2, 2], [[0, 1], [1, 0]]),
    ([1, 1], [[1, 3], [3, 6]]),
    ([2, 2, 1],
    [[1, 3, 0], [3, 6, 0], [0, 0, 1]]),
    ([2, 1, 1],
    [[-5, 0, 0], [0, 1, 3], [0, 3, 6]]),
] + [
    ([6, 6, 5, 5, 5], random_block_diagonal_symmetric_matrix(2, 3))
    for _ in range(10)
])
def test_simultaneous_diagonalize_real_symmetric_matrix_vs_singulars(
        s, m):
    m = np.array(m)
    s = np.diag(s)
    p = diagonalize.diagonalize_real_symmetric_and_sorted_diagonal_matrices(
        m, s)
    assertdiagonalized_by(s, p)
    assertdiagonalized_by(m, p)
    assert np.allclose(s, p.T.dot(s).dot(p))


@pytest.mark.parametrize('s,m', [
    ([1, 2], np.eye(2)),
    ([2, 1], [[0, 1], [1, 0]]),
    ([1, 0], [[1, 3], [3, 6]]),
    ([2, 1, 1], [[1, 3, 0], [3, 6, 0], [0, 0, 1]]),
    ([2, 2, 1], [[-5, 0, 0], [0, 1, 3], [0, 3, 6]]),
])
def test_simultaneous_diagonalize_real_symmetric_matrix_vs_singulars_fail(
        s, m):
    m = np.array(m)
    s = np.diag(s)
    with pytest.raises(ValueError):
        diagonalize.diagonalize_real_symmetric_and_sorted_diagonal_matrices(
            m, s)


@pytest.mark.parametrize('a,b', [
    (np.zeros((0, 0)), np.zeros((0, 0))),
    (np.eye(2), np.eye(2)),
    (np.eye(4), np.eye(4)),
    (np.eye(4), np.zeros((4, 4))),
    (H, H),
    (combinators.kron(np.eye(2), H),
     combinators.kron(H, np.eye(2))),
    (combinators.kron(np.eye(2), Z),
     combinators.kron(X, np.eye(2))),
] + [
    random_bi_diagonalizable_pair(2)
    for _ in range(10)
] + [
    random_bi_diagonalizable_pair(4)
    for _ in range(10)
] + [
    random_bi_diagonalizable_pair(4, d1, d2)
    for _ in range(10)
    for d1 in range(4)
    for d2 in range(4)
] + [
    random_bi_diagonalizable_pair(k)
    for k in range(1, 10)
])
def test_bidiagonalize_real_matrix_pair_with_symmetric_products(a, b):
    a = np.array(a)
    b = np.array(b)
    p, q = diagonalize.bidiagonalize_real_matrix_pair_with_symmetric_products(
        a, b)
    assert_bidiagonalized_by(a, p, q)
    assert_bidiagonalized_by(b, p, q)


@pytest.mark.parametrize('a,b', [
    [X, Z],
    [Y, np.eye(2)],
    [np.eye(2), Y],
    [np.eye(5), np.eye(4)],
    [
        e * testing.random_orthogonal(4)
        for e in random_bi_diagonalizable_pair(4)
        ],
    [
        testing.random_orthogonal(4) * e
        for e in random_bi_diagonalizable_pair(4)
    ],
])
def test_bidiagonalize_real_fails(a, b):
    a = np.array(a)
    b = np.array(b)
    with pytest.raises(ValueError):
        diagonalize.bidiagonalize_real_matrix_pair_with_symmetric_products(
            a, b)


@pytest.mark.parametrize('mat', [
    np.diag([1]),
    np.diag([1j]),
    np.diag([-1]),
    SWAP,
    CNOT,
    Y,
    H,
    combinators.kron(H, H),
    combinators.kron(Y, Y),
    QFT
] + [
    testing.random_unitary(2) for _ in
    range(10)
] + [
    testing.random_unitary(4) for _ in
    range(10)
] + [
    testing.random_unitary(k) for k in
    range(1, 10)
])
def test_bidiagonalize_unitary_with_special_orthogonals(mat):
    p, d, q = diagonalize.bidiagonalize_unitary_with_special_orthogonals(mat)
    assert predicates.is_special_orthogonal(p)
    assert predicates.is_special_orthogonal(q)
    assert np.allclose(p.dot(mat).dot(q), np.diag(d))
    assert_bidiagonalized_by(mat, p, q)


@pytest.mark.parametrize('mat', [
    np.diag([0]),
    np.diag([0.5]),
    np.diag([1, 0]),
    np.diag([0.5, 2]),
    np.array([[0, 1], [0, 0]]),
])
def test_bidiagonalize_unitary_fails(mat):
    with pytest.raises(ValueError):
        diagonalize.bidiagonalize_unitary_with_special_orthogonals(mat)
