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

from typing import Optional

import numpy as np
import pytest

import cirq

I2 = cirq.LinearOperator(np.eye(2), pauli_expansion=np.array([1., 0., 0., 0.]))
I4 = cirq.LinearOperator(np.eye(4))
I8 = cirq.LinearOperator(np.eye(8))
ZERO2 = cirq.LinearOperator(np.zeros((2, 2)),
                            pauli_expansion=np.array([0., 0., 0., 0.]))


@pytest.mark.parametrize('matrix, pauli_expansion', (
    (None, None),
    (np.ones((2, 2, 2)), None),
    (np.ones((2, 3)), None),
    (np.eye(4), np.array([1, 0, 0, 0])),
))
def test_init_invalid_inputs(matrix, pauli_expansion):
    with pytest.raises(ValueError):
        cirq.LinearOperator(matrix, pauli_expansion)


class FakeMatrixOperator(cirq.LinearOperator):
    def __init__(self, matrix: Optional[np.ndarray]) -> None:
        self._matrix = matrix
        self._pauli_expansion = None


@pytest.mark.parametrize('matrix, expected_pauli_expansion', (
    (None, None),
    (np.eye(2), np.array([1, 0, 0, 0])),
    (np.sqrt(2) * np.array([[1, 1], [1, -1]]),
     np.array([0, np.sqrt(2), 0, np.sqrt(2)])),
))
def test_pauli_expansion(matrix, expected_pauli_expansion):
    pauli_expansion = FakeMatrixOperator(matrix).pauli_expansion()
    assert ((pauli_expansion is None and expected_pauli_expansion is None) or
            np.all(pauli_expansion == expected_pauli_expansion))


class FakePauliExpansionOperator(cirq.LinearOperator):
    def __init__(self, pauli_expansion: Optional[np.array]) -> None:
        self._matrix = None
        self._pauli_expansion = pauli_expansion

    def _pauli_expansion_(self) -> Optional[np.ndarray]:
        return self._pauli_expansion


@pytest.mark.parametrize('pauli_expansion, expected_matrix', (
    (None, None),
    (np.array([1, 0, 0, 0]), np.eye(2)),
    (np.array([0, np.sqrt(2), 0, np.sqrt(2)]),
     np.sqrt(2) * np.array([[1, 1], [1, -1]])),
))
def test_matrix(pauli_expansion, expected_matrix):
    matrix = FakePauliExpansionOperator(pauli_expansion).matrix()
    assert ((matrix is None and expected_matrix is None) or
            np.all(matrix == expected_matrix))


@pytest.mark.parametrize('op', (
    cirq.X,
    cirq.H,
    -cirq.Y,
    2 * cirq.S,
    cirq.X + cirq.Y,
    cirq.X + cirq.Y,
    (cirq.X - cirq.Y)**4,
    (cirq.X - cirq.Y)**5,
    np.e**cirq.X,
    np.e**cirq.T,
    np.e**(-0.5j * np.pi * cirq.X),
    np.e**(-0.5j * np.pi * cirq.H),
    np.e**(-0.5j * np.pi * (cirq.Z - cirq.Z)),
    np.e**(-0.5j * np.pi * (cirq.Z + cirq.X)),
))
def test_internal_consistency(op):
    assert op.matrix() is not None
    assert op.pauli_expansion() is not None
    cirq.testing.assert_linear_operator_is_consistent(op)


@pytest.mark.parametrize('expression, expected_value', (
    (cirq.X + cirq.Z, cirq.H * np.sqrt(2)),
    (np.cos(0.1) * I2 - 1j * np.sin(0.1) * cirq.X, cirq.Rx(0.2)),
    (cirq.LinearOperator(np.array([[1, 2], [3, 4]])) +
     cirq.LinearOperator(np.array([[5, 3], [1, -1]])),
     cirq.LinearOperator(np.array([[6, 5], [4, 3]]))),
    (cirq.LinearOperator(pauli_expansion=np.array([1, 2, 3, 4])) +
     cirq.LinearOperator(pauli_expansion=np.array([5, 3, 1, -1])),
     cirq.LinearOperator(pauli_expansion=np.array([6, 5, 4, 3]))),
    (cirq.LinearOperator(np.array([[0, 1], [1, 0]])) +
     cirq.LinearOperator(pauli_expansion=np.array([0, 0, 0, 1])),
     cirq.LinearOperator(np.array([[1, 1], [1, -1]]))),
    (I4 + cirq.ZZ, cirq.LinearOperator(np.diag([2, 0, 0, 2]))),
    (cirq.TOFFOLI + cirq.TOFFOLI, 2 * cirq.TOFFOLI),
))
def test_addition(expression, expected_value):
    cirq.testing.assert_linear_operators_are_equal(expression, expected_value)


@pytest.mark.parametrize('expression, expected_value', (
    (cirq.X - cirq.X, ZERO2),
    (cirq.Y, 2 * cirq.Y - cirq.Y),
    (-cirq.Z + cirq.X, cirq.X - cirq.Z),
    (-cirq.X, cirq.LinearOperator(np.array([[0, -1], [-1, 0]]))),
    (-cirq.Ry(0.123), cirq.Ry(0.123 + 2 * np.pi)),
    (cirq.S - 1j * cirq.S, cirq.Rz(rads=np.pi / 2) * np.sqrt(2)),
    (-cirq.LinearOperator(np.array([[-1, 0], [1, 2]])),
     cirq.LinearOperator(np.array([[1, 0], [-1, -2]]))),
    (cirq.XX - cirq.YY,
     cirq.LinearOperator(np.rot90(np.diag([2, 0, 0, 2])))),
    (cirq.TOFFOLI - cirq.CCXPowGate(exponent=1),
     cirq.LinearOperator(np.zeros((8, 8)))),
))
def test_subtraction(expression, expected_value):
    cirq.testing.assert_linear_operators_are_equal(expression, expected_value)


@pytest.mark.parametrize('expression, expected_value', (
    (1.0 * cirq.X, cirq.X),
    (1j * cirq.H * 1j, -cirq.H),
    (-1j * cirq.Y, cirq.Ry(np.pi)),
    (np.sqrt(-1j) * cirq.S, cirq.Rz(np.pi / 2)),
    (2 * cirq.LinearOperator(np.array([[1, 2], [3, 4]])),
     cirq.LinearOperator(np.array([[2, 4], [6, 8]]))),
    ((cirq.X + cirq.Z) / np.sqrt(2), cirq.H),
    (0.5 * (I4 + cirq.XX + cirq.YY + cirq.ZZ), cirq.SWAP),
    (0.5 * (I4 + 1j * (cirq.XX + cirq.YY) + cirq.ZZ), cirq.ISWAP),
    (0 * cirq.CNOT, cirq.CZ * 0),
    (1 * cirq.CNOT + 0 * cirq.SWAP, cirq.CNOT),
    (0.5 * cirq.FREDKIN, cirq.FREDKIN / 2),
))
def test_scalar_multiplication(expression, expected_value):
    cirq.testing.assert_linear_operators_are_equal(expression, expected_value)


@pytest.mark.parametrize('expression, expected_value', (
    ((np.sqrt(1/3) * (cirq.X + cirq.Y + cirq.Z))**1,
     (np.sqrt(1/3) * (cirq.X + cirq.Y + cirq.Z))),
    ((np.sqrt(1/3) * (cirq.X + cirq.Y + cirq.Z))**3,
     (np.sqrt(1/3) * (cirq.X + cirq.Y + cirq.Z))),
    ((np.sqrt(1/3) * (cirq.X + cirq.Y + cirq.Z))**5,
     (np.sqrt(1/3) * (cirq.X + cirq.Y + cirq.Z))),
    ((np.sqrt(1/3) * (cirq.X + cirq.Y + cirq.Z))**7,
     (np.sqrt(1/3) * (cirq.X + cirq.Y + cirq.Z))),
    ((np.sqrt(1/3) * (cirq.X + cirq.Y + cirq.Z))**2, I2),
    ((np.sqrt(1/3) * (cirq.X + cirq.Y + cirq.Z))**4, I2),
    ((np.sqrt(1/3) * (cirq.X + cirq.Y + cirq.Z))**6, I2),

    ((np.sqrt(1/2) * cirq.H + np.sqrt(1/2) * cirq.Y)**157,
     (np.sqrt(1/2) * cirq.H + np.sqrt(1/2) * cirq.Y)),
    ((np.sqrt(1/2) * cirq.H + np.sqrt(1/2) * cirq.Y)**2019,
     (np.sqrt(1/2) * cirq.H + np.sqrt(1/2) * cirq.Y)),
    ((np.sqrt(1/2) * cirq.H + np.sqrt(1/2) * cirq.Y)**158, I2),
    ((np.sqrt(1/2) * cirq.H + np.sqrt(1/2) * cirq.Y)**2018, I2),

    ((I2 + cirq.X)**7,
     cirq.LinearOperator(np.full((2, 2), 64), np.array([64, 64, 0, 0]))),

    ((cirq.S + cirq.X)**2, cirq.Z + (1 + 1j) * cirq.X + I2),
    ((cirq.S + cirq.X)**3,
     (1 + 1j) * cirq.S**(-1) + (1 + 1j) * cirq.X + 2 * cirq.S),
))
def test_operator_power(expression, expected_value):
    cirq.testing.assert_linear_operators_are_equal(expression, expected_value)


@pytest.mark.parametrize('expression, expected_value', (
    (np.e**(-1.2j * cirq.X), cirq.Rx(2.4)),
    (np.exp(.5j * np.pi * cirq.Y), cirq.Ry(-np.pi)),
    (1j**cirq.Z, cirq.Rz(-np.pi)),
    (np.e**cirq.S, cirq.LinearOperator(np.diag([np.e, np.e**1j]))),
    (np.e**(.25j * np.pi * (cirq.XX + cirq.YY)), cirq.ISWAP),
    (np.e**(.125j * np.pi * (cirq.XX + cirq.YY)), cirq.ISWAP**0.5),
    (np.e**(.025j * np.pi * (cirq.XX + cirq.YY)), cirq.ISWAP**0.1),
    (np.e**(1j * np.pi * cirq.TOFFOLI / 2), 1j * cirq.TOFFOLI),
    (1j**cirq.LinearOperator(pauli_expansion=np.array([0, 1, 0, 1])),
     np.e**(.5j * np.pi * np.sqrt(2) * cirq.H)),
    (np.e**cirq.Z, cirq.LinearOperator(pauli_expansion=np.array(
        [(np.e + 1/np.e) / 2, 0, 0, (np.e - 1/np.e) / 2]))),
))
def test_operator_exponential(expression, expected_value):
    cirq.testing.assert_linear_operators_are_equal(expression, expected_value)


@pytest.mark.parametrize('expression, expected_unitary, expected_positive', (
    (cirq.X * 1j, 1j * cirq.X, I2),
    (cirq.X * 1e2, cirq.X, 1e2 * I2),
    (cirq.X * (-1e2), -cirq.X, 1e2 * I2),
    (cirq.X + cirq.Z, cirq.H, np.sqrt(2) * I2),
    (cirq.S, cirq.S, I2),
    (cirq.Y * 0, I2, ZERO2),
    (cirq.CNOT * 1e-6, cirq.CNOT, 1e-6 * I4),
    (cirq.TOFFOLI + cirq.TOFFOLI, cirq.TOFFOLI, 2 * I8),
    (cirq.LinearOperator(pauli_expansion=np.array([0, 1, 0, 1])),
     cirq.H, np.sqrt(2) * I2),
))
def test_polar_decomposition(expression, expected_unitary, expected_positive):
    unitary, positive = expression.polar_decomposition()
    cirq.testing.assert_linear_operators_are_equal(unitary, expected_unitary)
    cirq.testing.assert_linear_operators_are_equal(positive, expected_positive)

    unitary2 = expression.unitary_factor()
    cirq.testing.assert_linear_operators_are_equal(unitary2, expected_unitary)


@pytest.mark.parametrize('expression', (
    cirq.X, 2 * cirq.Y, cirq.X + cirq.Z, 0 * cirq.S, -cirq.T,
    np.e**(-.25j * np.pi * cirq.X), (cirq.X + 0.5 * cirq.Z)**17,
    (cirq.X + cirq.Y).unitary_factor(),
))
def test_can_compute_both_matrix_and_pauli_expansion(expression):
    assert expression.matrix() is not None
    assert expression.pauli_expansion() is not None
    cirq.testing.assert_linear_operator_is_consistent(expression)


@pytest.mark.parametrize('expression, string', (
    (0 * cirq.X, '0'),
    (cirq.Y * 2, '2.000*Y'),
    (cirq.X - 1j * cirq.Y, '1.000*X-1.000i*Y'),
    ((cirq.X + cirq.Y + cirq.Z) / np.sqrt(3), '0.577*X+0.577*Y+0.577*Z'),
    (np.sqrt(1j) * cirq.X**0, '(0.707+0.707i)*I'),
    (np.sqrt(-1j) * cirq.X**1, '(0.707-0.707i)*X'),
    (2 * cirq.CNOT, '[[2.+0.j 0.+0.j 0.+0.j 0.+0.j]\n'
                    ' [0.+0.j 2.+0.j 0.+0.j 0.+0.j]\n'
                    ' [0.+0.j 0.+0.j 0.+0.j 2.+0.j]\n'
                    ' [0.+0.j 0.+0.j 2.+0.j 0.+0.j]]'),
))
def test_operator_str(expression, string):
    assert str(expression) == string


@pytest.mark.parametrize('original', (
    .5 * cirq.X, cirq.X + cirq.Y, cirq.CZ / 2, np.exp(1j * cirq.X * np.pi),
    cirq.LinearOperator(np.eye(2)),
    cirq.LinearOperator(pauli_expansion=np.array([0, 1, 0, 1])),
    cirq.LinearOperator(matrix=np.diag([1, -1]),
                        pauli_expansion=np.array([0, 0, 0, 1])),
))
def test_operator_repr(original):
    recovered = eval(repr(original))
    cirq.testing.assert_linear_operators_are_equal(recovered, original)
