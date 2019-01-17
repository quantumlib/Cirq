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
import re

import numpy as np
import pytest

import cirq

H = np.array([[1, 1], [1, -1]]) * np.sqrt(0.5)
HH = cirq.kron(H, H)
QFT2 = np.array([[1, 1, 1, 1],
                 [1, 1j, -1, -1j],
                 [1, -1, 1, -1],
                 [1, -1j, -1, 1j]]) * 0.5


def test_single_qubit_init():
    m = np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5)
    x2 = cirq.SingleQubitMatrixGate(m)
    assert cirq.has_unitary(x2)
    assert np.alltrue(cirq.unitary(x2) == m)


def test_single_qubit_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.SingleQubitMatrixGate(np.eye(2)))
    eq.make_equality_group(
        lambda: cirq.SingleQubitMatrixGate(np.array([[0, 1], [1, 0]])))
    x2 = np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5)
    eq.make_equality_group(lambda: cirq.SingleQubitMatrixGate(x2))


def test_single_qubit_trace_distance_bound():
    x = cirq.SingleQubitMatrixGate(np.array([[0, 1], [1, 0]]))
    x2 = cirq.SingleQubitMatrixGate(
        np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5))
    assert cirq.trace_distance_bound(x) >= 1
    assert cirq.trace_distance_bound(x2) >= 0.5


def test_single_qubit_approx_eq():
    x = cirq.SingleQubitMatrixGate(np.array([[0, 1], [1, 0]]))
    i = cirq.SingleQubitMatrixGate(np.array([[1, 0], [0, 1]]))
    i_ish = cirq.SingleQubitMatrixGate(
        np.array([[1, 0.000000000000001], [0, 1]]))
    assert i.approx_eq(i_ish)
    assert i.approx_eq(i)
    assert not i.approx_eq(x)
    assert i.approx_eq('') is NotImplemented


def test_single_qubit_extrapolate():
    i = cirq.SingleQubitMatrixGate(np.eye(2))
    x = cirq.SingleQubitMatrixGate(np.array([[0, 1], [1, 0]]))
    x2 = cirq.SingleQubitMatrixGate(
        np.array([[1, 1j], [1j, 1]]) * (1 - 1j) / 2)
    assert cirq.has_unitary(x2)
    x2i = cirq.SingleQubitMatrixGate(np.conj(cirq.unitary(x2).T))

    assert (x**0).approx_eq(i)
    assert (x2**0).approx_eq(i)
    assert (x2**2).approx_eq(x)
    assert (x2**-1).approx_eq(x2i)
    assert (x2**3).approx_eq(x2i)
    assert (x**-1).approx_eq(x)

    z2 = cirq.SingleQubitMatrixGate(np.array([[1, 0], [0, 1j]]))
    z4 = cirq.SingleQubitMatrixGate(
        np.array([[1, 0], [0, (1 + 1j) * np.sqrt(0.5)]]))
    assert (z2**0.5).approx_eq(z4)
    with pytest.raises(TypeError):
        _ = x**cirq.Symbol('a')


def test_two_qubit_init():
    x2 = cirq.TwoQubitMatrixGate(QFT2)
    assert cirq.has_unitary(x2)
    assert np.alltrue(cirq.unitary(x2) == QFT2)


def test_two_qubit_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.TwoQubitMatrixGate(np.eye(4)))
    eq.make_equality_group(lambda: cirq.TwoQubitMatrixGate(QFT2))
    eq.make_equality_group(lambda: cirq.TwoQubitMatrixGate(HH))


def test_two_qubit_approx_eq():
    f = cirq.TwoQubitMatrixGate(QFT2)
    perturb = np.zeros(shape=QFT2.shape, dtype=np.float64)
    perturb[1, 2] = 0.00000001
    assert f.approx_eq(cirq.TwoQubitMatrixGate(QFT2))
    assert f.approx_eq(cirq.TwoQubitMatrixGate(QFT2 + perturb))
    assert not f.approx_eq(cirq.TwoQubitMatrixGate(HH))


def test_two_qubit_extrapolate():
    cz2 = cirq.TwoQubitMatrixGate(np.diag([1, 1, 1, 1j]))
    cz4 = cirq.TwoQubitMatrixGate(np.diag([1, 1, 1, (1 + 1j) * np.sqrt(0.5)]))
    i = cirq.TwoQubitMatrixGate(np.eye(4))

    assert (cz2**0).approx_eq(i)
    assert (cz4**0).approx_eq(i)
    assert (cz2**0.5).approx_eq(cz4)
    with pytest.raises(TypeError):
        _ = cz2**cirq.Symbol('a')


def test_single_qubit_diagram():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    m = np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5)
    c = cirq.Circuit.from_ops(
        cirq.SingleQubitMatrixGate(m).on(a),
        cirq.CZ(a, b))

    assert re.match("""
a: ───┌[            ]+┐───@───
      │[0-9\\.+\\-j ]+│   │
      │[0-9\\.+\\-j ]+│   │
      └[            ]+┘   │
       [            ]+    │
b: ────[────────────]+────@───
    """.strip(), c.to_text_diagram())

    assert re.match(r"""
a: ---[\[0-9\.+\-j \]]+---@---
      [\[0-9\.+\-j \]]+   |
      [              ]+   |
b: ---[--------------]+---@---
        """.strip(), c.to_text_diagram(use_unicode_characters=False))

    assert re.match("""
a[            ]+  b
│[            ]+  │
┌[            ]+┐ │
│[0-9\\.+\\-j ]+│ │
│[0-9\\.+\\-j ]+│ │
└[            ]+┘ │
│[            ]+  │
@[────────────]+──@
│[            ]+  │
    """.strip(), c.to_text_diagram(transpose=True))


def test_two_qubit_diagram():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    c = cirq.Circuit.from_ops(
        cirq.TwoQubitMatrixGate(cirq.unitary(cirq.CZ)).on(a, b),
        cirq.TwoQubitMatrixGate(cirq.unitary(cirq.CZ)).on(c, a))
    assert re.match("""
a: ───┌[            ]+┐───#2─+
      │[0-9\\.+\\-j ]+│   │
      │[0-9\\.+\\-j ]+│   │
      │[0-9\\.+\\-j ]+│   │
      │[0-9\\.+\\-j ]+│   │
      └[            ]+┘   │
      │[            ]+    │
b: ───#2[───────────]+────┼──+
       [            ]+    │
c: ────[────────────]+────┌[            ]+┐───
       [            ]+    │[0-9\\.+\\-j ]+│
       [            ]+    │[0-9\\.+\\-j ]+│
       [            ]+    │[0-9\\.+\\-j ]+│
       [            ]+    │[0-9\\.+\\-j ]+│
       [            ]+    └[            ]+┘
    """.strip(), c.to_text_diagram())

    assert re.match("""
a[            ]+  b  c
│[            ]+  │  │
┌[            ]+┐─#2 │
│[0-9\\.+\\-j ]+│ │  │
│[0-9\\.+\\-j ]+│ │  │
│[0-9\\.+\\-j ]+│ │  │
│[0-9\\.+\\-j ]+│ │  │
└[            ]+┘ │  │
│[            ]+  │  │
#2[───────────]+──┼──┌[            ]+┐
│[            ]+  │  │[0-9\\.+\\-j ]+│
│[            ]+  │  │[0-9\\.+\\-j ]+│
│[            ]+  │  │[0-9\\.+\\-j ]+│
│[            ]+  │  │[0-9\\.+\\-j ]+│
│[            ]+  │  └[            ]+┘
│[            ]+  │  │
    """.strip(), c.to_text_diagram(transpose=True))


def test_str_executes():
    assert '1' in str(cirq.SingleQubitMatrixGate(np.eye(2)))
    assert '0' in str(cirq.TwoQubitMatrixGate(np.eye(4)))


def test_one_qubit_consistent():
    u = cirq.testing.random_unitary(2)
    g = cirq.SingleQubitMatrixGate(u)
    cirq.testing.assert_implements_consistent_protocols(g)


def test_two_qubit_consistent():
    u = cirq.testing.random_unitary(4)
    g = cirq.TwoQubitMatrixGate(u)
    cirq.testing.assert_implements_consistent_protocols(g)


@pytest.mark.parametrize('expression, expected_matrix', (
    (-cirq.X,
     cirq.SingleQubitMatrixGate(np.array([[0, -1], [-1, 0]]))),
    (1j * cirq.H,
     cirq.SingleQubitMatrixGate(1j * np.sqrt(.5) * np.array([[1, 1],
                                                             [1, -1]]))),
    (cirq.S / 1j,
     cirq.SingleQubitMatrixGate(np.diag([-1j, 1]))),
    ((cirq.X + cirq.Y) / np.sqrt(2),
     cirq.SingleQubitMatrixGate(np.array([[0, np.sqrt(-1j)],
                                          [np.sqrt(1j), 0]]))),
    ((cirq.X - cirq.Y) / np.sqrt(2),
     cirq.SingleQubitMatrixGate(np.array([[0, np.sqrt(1j)],
                                          [np.sqrt(-1j), 0]]))),
    (np.e**(-.25j * cirq.Y * np.pi),
     cirq.SingleQubitMatrixGate(np.sqrt(.5) * np.array([[1, -1], [1, 1]]))),
    (-cirq.CZ,
     cirq.TwoQubitMatrixGate(np.diag([-1, -1, -1, 1]))),
    (2**(cirq.CNOT - cirq.CNOT),
     cirq.TwoQubitMatrixGate(np.eye(4))),
    (np.exp(1j * np.pi * (cirq.XX + cirq.YY) / 4),
     cirq.TwoQubitMatrixGate(np.array([[1, 0, 0, 0],
                                       [0, 0, 1j, 0],
                                       [0, 1j, 0, 0],
                                       [0, 0, 0, 1]]))),
))
def test_make_gate(expression, expected_matrix):
    assert cirq.make_gate(expression).approx_eq(expected_matrix,
                                                ignore_global_phase=False)


class FakeLinearOperator(cirq.AbstractLinearOperator):
    def __init__(self, matrix: Optional[np.ndarray]) -> None:
        self._matrix = matrix

    def _n_dim_(self) -> Optional[int]:
        pass

    def _matrix_(self) -> Optional[np.ndarray]:
        return self._matrix

    def _pauli_expansion_(self) -> None:
        pass


@pytest.mark.parametrize('expression', (
    2 * cirq.Y, cirq.X + cirq.Z, cirq.CNOT + cirq.CZ, cirq.TOFFOLI,
    FakeLinearOperator(None),
    FakeLinearOperator(np.zeros((2, 2, 2))),
    FakeLinearOperator(np.zeros((2, 3))),
    FakeLinearOperator(np.zeros((3, 3))),
))
def test_make_gate_errors(expression):
    with pytest.raises(ValueError):
        cirq.make_gate(expression)


@pytest.mark.parametrize('expression, expected_matrix', (
    (2 * cirq.Y,
     cirq.SingleQubitMatrixGate(np.array([[0, -1j], [1j, 0]]))),
    (cirq.X + cirq.Z,
     cirq.SingleQubitMatrixGate(np.sqrt(.5) * np.array([[1, 1],
                                                        [1, -1]]))),
    (3j * cirq.H,
     cirq.SingleQubitMatrixGate(1j * np.sqrt(.5) * np.array([[1, 1],
                                                             [1, -1]]))),
    (cirq.S / 10j,
     cirq.SingleQubitMatrixGate(np.diag([-1j, 1]))),
    (cirq.X + cirq.Y,
     cirq.SingleQubitMatrixGate(np.array([[0, np.sqrt(-1j)],
                                          [np.sqrt(1j), 0]]))),
    (cirq.X - cirq.Y,
     cirq.SingleQubitMatrixGate(np.array([[0, np.sqrt(1j)],
                                          [np.sqrt(-1j), 0]]))),
    (np.e**cirq.Y,
     cirq.SingleQubitMatrixGate(np.eye(2))),
    (cirq.CNOT + cirq.CZ,
     cirq.TwoQubitMatrixGate(np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, np.sqrt(.5), np.sqrt(.5)],
                                       [0, 0, np.sqrt(.5), -np.sqrt(.5)]]))),
    (-2 * cirq.CZ,
     cirq.TwoQubitMatrixGate(np.diag([-1, -1, -1, 1]))),
    (2**cirq.CNOT,
     cirq.TwoQubitMatrixGate(np.eye(4))),
    (2 * np.exp(1j * np.pi * (cirq.XX + cirq.YY) / 4),
     cirq.TwoQubitMatrixGate(np.array([[1, 0, 0, 0],
                                       [0, 0, 1j, 0],
                                       [0, 1j, 0, 0],
                                       [0, 0, 0, 1]]))),
))
def test_forge_gate(expression, expected_matrix):
    assert cirq.forge_gate(expression).approx_eq(expected_matrix,
                                                 ignore_global_phase=False)


@pytest.mark.parametrize('expression', (
    cirq.TOFFOLI, cirq.FREDKIN * 1j,
))
def test_forge_gate_errors(expression):
    with pytest.raises(ValueError):
        cirq.forge_gate(expression)
