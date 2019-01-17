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
STANDARD_BASIS = (E00, E01, E10, E11)


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
    (STANDARD_BASIS, cirq.linalg.operator_spaces.PAULI_BASIS),
))
def test_expand_in_basis_reconstruction(m, basis):
    coefficients = cirq.linalg.expand_in_basis(m, basis)
    reconstructed = np.zeros(m.shape, dtype=complex)
    for coefficient, element in zip(coefficients, basis):
        reconstructed += coefficient * element
    assert np.allclose(m, reconstructed)


@pytest.mark.parametrize(
    'expansion,basis',
    itertools.product((
        np.array([1, 0, 0, 0]),
        np.array([0, 1, 0, 0]),
        np.array([0, 0, 1, 0]),
        np.array([0, 0, 0, 1]),
        np.array([0, 1, 0, 1]),
        np.array([0.5, 0.4, 0.3, 0.2]),
        np.array([1, 2, 3, 4]),
    ), (STANDARD_BASIS, cirq.linalg.operator_spaces.PAULI_BASIS)),
)
def test_reconstruct_from_expansion_expand(expansion, basis):
    m = cirq.linalg.reconstruct_from_expansion(expansion, basis)
    for coefficient, element in zip(expansion, basis):
        expected_coefficient = (cirq.linalg.hilbert_schmidt(m, element) /
                                cirq.linalg.hilbert_schmidt(element, element))
        assert np.isclose(coefficient, expected_coefficient)


@pytest.mark.parametrize(
    'm1,basis', (
    itertools.product(
        (I, X, Y, Z, H, SQRT_X, SQRT_Y, SQRT_Z, E00, E01, E10, E11),
        (STANDARD_BASIS, cirq.linalg.operator_spaces.PAULI_BASIS),
    )
))
def test_expand_in_basis_is_inverse_of_reconstruct_from_expansion(m1, basis):
    c1 = cirq.linalg.expand_in_basis(m1, basis)
    m2 = cirq.linalg.reconstruct_from_expansion(c1, basis)
    c2 = cirq.linalg.expand_in_basis(m2, basis)
    assert np.allclose(m1, m2)
    assert np.allclose(c1, c2)


@pytest.mark.parametrize(
    'coefficients,exponent',
    itertools.product((
        np.array([0, 0, 0, 0]),
        np.array([0.5, 0, 0, 0]),
        np.array([1, 0, 0, 0]),
        np.array([2, 0, 0, 0]),
        np.array([0, 0.5, 0, 0]),
        np.array([0, 1, 0, 0]),
        np.array([0, 2, 0, 0]),
        np.array([0, 0, 0.5, 0]),
        np.array([0, 0, 1, 0]),
        np.array([0, 0, 2, 0]),
        np.array([0, 0, 0, 0.5]),
        np.array([0, 0, 0, 1]),
        np.array([0, 0, 0, 2]),
        np.array([0, 0.5, 0, 0.5]),
        np.array([0, 1, 0, 1]),
        np.array([0, 2, 0, 2]),
        np.array([0, 1, -1j, 0]),
        np.array([0.25j, 0.25j, 0.25j, 0.25j]),
        np.array([0.25, 0.25, 0.25, 0.25]),
        np.array([0.4, 0, 0.6, 0]),
        np.array([1, 2, 3, 4]),
    ), (0, 1, 2, 3, 4, 5, 10, 11, 20, 21, 100, 101)),
)
def test_operator_power(coefficients, exponent):
    i, x, y, z = cirq.linalg.operator_spaces.PAULI_BASIS

    a, b, c, d = coefficients
    matrix = a * i + b * x + c * y + d * z
    expected_result = np.linalg.matrix_power(matrix, exponent)

    a2, b2, c2, d2 = cirq.linalg.operator_power(coefficients, exponent)
    result = a2 * i + b2 * x + c2 * y + d2 * z

    assert np.allclose(result, expected_result)
