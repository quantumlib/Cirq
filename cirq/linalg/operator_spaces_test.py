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

import itertools

import numpy as np
import pytest
import scipy.linalg

import cirq

I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = np.array([[1, 1], [1, -1]]) * np.sqrt(0.5)
SQRT_X = np.array([[np.sqrt(1j), np.sqrt(-1j)],
                   [np.sqrt(-1j), np.sqrt(1j)]]) * np.sqrt(0.5)
SQRT_Y = np.array([[np.sqrt(1j), -np.sqrt(1j)],
                   [np.sqrt(1j), np.sqrt(1j)]]) * np.sqrt(0.5)
SQRT_Z = np.diag([1, 1j])
E00 = np.diag([1, 0])
E01 = np.array([[0, 1], [0, 0]])
E10 = np.array([[0, 0], [1, 0]])
E11 = np.diag([0, 1])
PAULI_BASIS = cirq.linalg.operator_spaces.PAULI_BASIS
STANDARD_BASIS = {'a': E00, 'b': E01, 'c': E10, 'd': E11}


def _one_hot_matrix(size: int, i: int, j: int) -> np.ndarray:
    result = np.zeros((size, size))
    result[i, j] = 1
    return result


@pytest.mark.parametrize('basis1, basis2, expected_kron_basis', (
    (PAULI_BASIS, PAULI_BASIS, {
        'II': np.eye(4),
        'IX': scipy.linalg.block_diag(X, X),
        'IY': scipy.linalg.block_diag(Y, Y),
        'IZ': np.diag([1, -1, 1, -1]),
        'XI': np.array([[0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0]]),
        'XX': np.rot90(np.eye(4)),
        'XY': np.rot90(np.diag([1j, -1j, 1j, -1j])),
        'XZ': np.array([[0, 0, 1, 0],
                        [0, 0, 0, -1],
                        [1, 0, 0, 0],
                        [0, -1, 0, 0]]),
        'YI': np.array([[0, 0, -1j, 0],
                        [0, 0, 0, -1j],
                        [1j, 0, 0, 0],
                        [0, 1j, 0, 0]]),
        'YX': np.rot90(np.diag([1j, 1j, -1j, -1j])),
        'YY': np.rot90(np.diag([-1, 1, 1, -1])),
        'YZ': np.array([[0, 0, -1j, 0],
                        [0, 0, 0, 1j],
                        [1j, 0, 0, 0],
                        [0, -1j, 0, 0]]),
        'ZI': np.diag([1, 1, -1, -1]),
        'ZX': scipy.linalg.block_diag(X, -X),
        'ZY': scipy.linalg.block_diag(Y, -Y),
        'ZZ': np.diag([1, -1, -1, 1]),
    }),
    (STANDARD_BASIS, STANDARD_BASIS, {
        'aa': _one_hot_matrix(4, 0, 0),
        'ab': _one_hot_matrix(4, 0, 1),
        'ac': _one_hot_matrix(4, 1, 0),
        'ad': _one_hot_matrix(4, 1, 1),
        'ba': _one_hot_matrix(4, 0, 2),
        'bb': _one_hot_matrix(4, 0, 3),
        'bc': _one_hot_matrix(4, 1, 2),
        'bd': _one_hot_matrix(4, 1, 3),
        'ca': _one_hot_matrix(4, 2, 0),
        'cb': _one_hot_matrix(4, 2, 1),
        'cc': _one_hot_matrix(4, 3, 0),
        'cd': _one_hot_matrix(4, 3, 1),
        'da': _one_hot_matrix(4, 2, 2),
        'db': _one_hot_matrix(4, 2, 3),
        'dc': _one_hot_matrix(4, 3, 2),
        'dd': _one_hot_matrix(4, 3, 3),
    }),
))
def test_kron_bases(basis1, basis2, expected_kron_basis):
    kron_basis = cirq.linalg.kron_bases(basis1, basis2)
    assert len(kron_basis) == 16
    assert kron_basis.keys() == expected_kron_basis.keys()
    for name in kron_basis.keys():
        assert np.all(kron_basis[name] == expected_kron_basis[name])


@pytest.mark.parametrize('m1,m2,expect_real', (
    (X, X, True),
    (X, Y, True),
    (X, H, True),
    (X, SQRT_X, False),
    (I, SQRT_Z, False),
))
def test_hilbert_schmidt_is_conjugate_symmetric(m1, m2, expect_real):
    v1 = cirq.linalg.hilbert_schmidt(m1, m2)
    v2 = cirq.linalg.hilbert_schmidt(m2, m1)
    assert v1 == v2.conjugate()

    assert np.isreal(v1) == expect_real
    if not expect_real:
        assert v1 != v2


@pytest.mark.parametrize('a,m1,b,m2', (
    (1, X, 1, Z),
    (2, X, 3, Y),
    (2j, X, 3, I),
    (2, X, 3, X),
))
def test_hilbert_schmidt_is_linear(a, m1, b, m2):
    v1 = cirq.linalg.hilbert_schmidt(H, (a * m1 + b * m2))
    v2 = (a * cirq.linalg.hilbert_schmidt(H, m1) +
          b * cirq.linalg.hilbert_schmidt(H, m2))
    assert v1 == v2


@pytest.mark.parametrize('m', (I, X, Y, Z, H, SQRT_X, SQRT_Y, SQRT_Z))
def test_hilbert_schmidt_is_positive_definite(m):
    v = cirq.linalg.hilbert_schmidt(m, m)
    assert np.isreal(v)
    assert v.real > 0


@pytest.mark.parametrize('m1,m2,expected_value', (
    (X, I, 0),
    (X, X, 2),
    (X, Y, 0),
    (X, Z, 0),
    (H, X, np.sqrt(2)),
    (H, Y, 0),
    (H, Z, np.sqrt(2)),
    (Z, E00, 1),
    (Z, E01, 0),
    (Z, E10, 0),
    (Z, E11, -1),
    (SQRT_X, E00, np.sqrt(-.5j)),
    (SQRT_X, E01, np.sqrt(.5j)),
    (SQRT_X, E10, np.sqrt(.5j)),
    (SQRT_X, E11, np.sqrt(-.5j)),
))
def test_hilbert_schmidt_values(m1, m2, expected_value):
    v = cirq.linalg.hilbert_schmidt(m1, m2)
    assert np.isclose(v, expected_value)


@pytest.mark.parametrize('m,basis', itertools.product(
    (I, X, Y, Z, H, SQRT_X, SQRT_Y, SQRT_Z),
    (PAULI_BASIS, STANDARD_BASIS),
))
def test_expand_in_basis(m, basis):
    expansion = cirq.linalg.expand_in_basis(m, basis)

    reconstructed = np.zeros(m.shape, dtype=complex)
    for name, coefficient in expansion.items():
        reconstructed += coefficient * basis[name]
    assert np.allclose(m, reconstructed)


@pytest.mark.parametrize('expansion', (
    {'I': 1}, {'X': 1}, {'Y': 1}, {'Z': 1}, {'X': 1, 'Z': 1},
    {'I': 0.5, 'X': 0.4, 'Y': 0.3, 'Z': 0.2},
    {'I': 1, 'X': 2, 'Y': 3, 'Z': 4},
))
def test_reconstruct_from_expansion(expansion):
    m = cirq.linalg.reconstruct_from_expansion(expansion, PAULI_BASIS)

    for name, coefficient in expansion.items():
        element = PAULI_BASIS[name]
        expected_coefficient = (cirq.linalg.hilbert_schmidt(m, element) /
                                cirq.linalg.hilbert_schmidt(element, element))
        assert np.isclose(coefficient, expected_coefficient)


@pytest.mark.parametrize(
    'm1,basis', (
    itertools.product(
        (I, X, Y, Z, H, SQRT_X, SQRT_Y, SQRT_Z, E00, E01, E10, E11),
        (PAULI_BASIS, STANDARD_BASIS),
    )
))
def test_expand_in_basis_is_inverse_of_reconstruct_from_expansion(m1, basis):
    c1 = cirq.linalg.expand_in_basis(m1, basis)
    m2 = cirq.linalg.reconstruct_from_expansion(c1, basis)
    c2 = cirq.linalg.expand_in_basis(m2, basis)
    assert np.allclose(m1, m2)
    assert c1 == c2
