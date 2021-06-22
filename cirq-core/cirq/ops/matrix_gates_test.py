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
import re

import numpy as np
import pytest
import sympy

import cirq

H = np.array([[1, 1], [1, -1]]) * np.sqrt(0.5)
HH = cirq.kron(H, H)
QFT2 = np.array([[1, 1, 1, 1], [1, 1j, -1, -1j], [1, -1, 1, -1], [1, -1j, -1, 1j]]) * 0.5
PLUS_ONE = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])


def test_single_qubit_init():
    m = np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5)
    x2 = cirq.MatrixGate(m)
    assert cirq.has_unitary(x2)
    assert np.alltrue(cirq.unitary(x2) == m)
    assert cirq.qid_shape(x2) == (2,)

    x2 = cirq.MatrixGate(PLUS_ONE, qid_shape=(3,))
    assert cirq.has_unitary(x2)
    assert np.alltrue(cirq.unitary(x2) == PLUS_ONE)
    assert cirq.qid_shape(x2) == (3,)

    with pytest.raises(ValueError, match='Not a .*unitary matrix'):
        cirq.MatrixGate(np.zeros((2, 2)))
    with pytest.raises(ValueError, match='must be a square 2d numpy array'):
        cirq.MatrixGate(cirq.eye_tensor((2, 2), dtype=float))
    with pytest.raises(ValueError, match='must be a square 2d numpy array'):
        cirq.MatrixGate(np.ones((3, 4)))
    with pytest.raises(ValueError, match='must be a square 2d numpy array'):
        cirq.MatrixGate(np.ones((2, 2, 2)))


def test_single_qubit_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.MatrixGate(np.eye(2)))
    eq.make_equality_group(lambda: cirq.MatrixGate(np.array([[0, 1], [1, 0]])))
    x2 = np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5)
    eq.make_equality_group(lambda: cirq.MatrixGate(x2))
    eq.add_equality_group(cirq.MatrixGate(PLUS_ONE, qid_shape=(3,)))


def test_single_qubit_trace_distance_bound():
    x = cirq.MatrixGate(np.array([[0, 1], [1, 0]]))
    x2 = cirq.MatrixGate(np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5))
    assert cirq.trace_distance_bound(x) >= 1
    assert cirq.trace_distance_bound(x2) >= 0.5


def test_single_qubit_approx_eq():
    x = cirq.MatrixGate(np.array([[0, 1], [1, 0]]))
    i = cirq.MatrixGate(np.array([[1, 0], [0, 1]]))
    i_ish = cirq.MatrixGate(np.array([[1, 0.000000000000001], [0, 1]]))
    assert cirq.approx_eq(i, i_ish, atol=1e-9)
    assert cirq.approx_eq(i, i, atol=1e-9)
    assert not cirq.approx_eq(i, x, atol=1e-9)
    assert not cirq.approx_eq(i, '', atol=1e-9)


def test_single_qubit_extrapolate():
    i = cirq.MatrixGate(np.eye(2))
    x = cirq.MatrixGate(np.array([[0, 1], [1, 0]]))
    x2 = cirq.MatrixGate(np.array([[1, 1j], [1j, 1]]) * (1 - 1j) / 2)
    assert cirq.has_unitary(x2)
    x2i = cirq.MatrixGate(np.conj(cirq.unitary(x2).T))

    assert cirq.approx_eq(x ** 0, i, atol=1e-9)
    assert cirq.approx_eq(x2 ** 0, i, atol=1e-9)
    assert cirq.approx_eq(x2 ** 2, x, atol=1e-9)
    assert cirq.approx_eq(x2 ** -1, x2i, atol=1e-9)
    assert cirq.approx_eq(x2 ** 3, x2i, atol=1e-9)
    assert cirq.approx_eq(x ** -1, x, atol=1e-9)

    z2 = cirq.MatrixGate(np.array([[1, 0], [0, 1j]]))
    z4 = cirq.MatrixGate(np.array([[1, 0], [0, (1 + 1j) * np.sqrt(0.5)]]))
    assert cirq.approx_eq(z2 ** 0.5, z4, atol=1e-9)
    with pytest.raises(TypeError):
        _ = x ** sympy.Symbol('a')


def test_two_qubit_init():
    x2 = cirq.MatrixGate(QFT2)
    assert cirq.has_unitary(x2)
    assert np.alltrue(cirq.unitary(x2) == QFT2)


def test_two_qubit_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.MatrixGate(np.eye(4)))
    eq.make_equality_group(lambda: cirq.MatrixGate(QFT2))
    eq.make_equality_group(lambda: cirq.MatrixGate(HH))


def test_two_qubit_approx_eq():
    f = cirq.MatrixGate(QFT2)
    perturb = np.zeros(shape=QFT2.shape, dtype=np.float64)
    perturb[1, 2] = 1e-8

    assert cirq.approx_eq(f, cirq.MatrixGate(QFT2), atol=1e-9)

    assert not cirq.approx_eq(f, cirq.MatrixGate(QFT2 + perturb), atol=1e-9)
    assert cirq.approx_eq(f, cirq.MatrixGate(QFT2 + perturb), atol=1e-7)

    assert not cirq.approx_eq(f, cirq.MatrixGate(HH), atol=1e-9)


def test_two_qubit_extrapolate():
    cz2 = cirq.MatrixGate(np.diag([1, 1, 1, 1j]))
    cz4 = cirq.MatrixGate(np.diag([1, 1, 1, (1 + 1j) * np.sqrt(0.5)]))
    i = cirq.MatrixGate(np.eye(4))

    assert cirq.approx_eq(cz2 ** 0, i, atol=1e-9)
    assert cirq.approx_eq(cz4 ** 0, i, atol=1e-9)
    assert cirq.approx_eq(cz2 ** 0.5, cz4, atol=1e-9)
    with pytest.raises(TypeError):
        _ = cz2 ** sympy.Symbol('a')


def test_single_qubit_diagram():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    m = np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5)
    c = cirq.Circuit(cirq.MatrixGate(m).on(a), cirq.CZ(a, b))

    assert re.match(
        r"""
      ┌[          ]+┐
a: ───│[0-9\.+\-j ]+│───@───
      │[0-9\.+\-j ]+│   │
      └[          ]+┘   │
       [          ]+    │
b: ────[──────────]+────@───
    """.strip(),
        c.to_text_diagram().strip(),
    )

    assert re.match(
        r"""
a[          ]+  b
│[          ]+  │
┌[          ]+┐ │
│[0-9\.+\-j ]+│ │
│[0-9\.+\-j ]+│ │
└[          ]+┘ │
│[          ]+  │
@[──────────]+──@
│[          ]+  │
    """.strip(),
        c.to_text_diagram(transpose=True).strip(),
    )


def test_two_qubit_diagram():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    c = cirq.Circuit(
        cirq.MatrixGate(cirq.unitary(cirq.CZ)).on(a, b),
        cirq.MatrixGate(cirq.unitary(cirq.CZ)).on(c, a),
    )
    assert re.match(
        r"""
      ┌[          ]+┐
      │[0-9\.+\-j ]+│
a: ───│[0-9\.+\-j ]+│───#2─+
      │[0-9\.+\-j ]+│   │
      │[0-9\.+\-j ]+│   │
      └[          ]+┘   │
      │[          ]+    │
b: ───#2[─────────]+────┼──+
       [          ]+    │
       [          ]+    ┌[          ]+┐
       [          ]+    │[0-9\.+\-j ]+│
c: ────[──────────]+────│[0-9\.+\-j ]+│──+
       [          ]+    │[0-9\.+\-j ]+│
       [          ]+    │[0-9\.+\-j ]+│
       [          ]+    └[          ]+┘
    """.strip(),
        c.to_text_diagram().strip(),
    )

    assert re.match(
        r"""
a[          ]+  b  c
│[          ]+  │  │
┌[          ]+┐ │  │
│[0-9\.+\-j ]+│ │  │
│[0-9\.+\-j ]+│─#2 │
│[0-9\.+\-j ]+│ │  │
│[0-9\.+\-j ]+│ │  │
└[          ]+┘ │  │
│[          ]+  │  │
│[          ]+  │  ┌[          ]+┐
│[          ]+  │  │[0-9\.+\-j ]+│
#2[─────────]+──┼──│[0-9\.+\-j ]+│
│[          ]+  │  │[0-9\.+\-j ]+│
│[          ]+  │  │[0-9\.+\-j ]+│
│[          ]+  │  └[          ]+┘
│[          ]+  │  │
    """.strip(),
        c.to_text_diagram(transpose=True).strip(),
    )


def test_named_single_qubit_diagram():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    m = np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5)
    c = cirq.Circuit(cirq.MatrixGate(m, name='Foo').on(a), cirq.CZ(a, b))

    expected_horizontal = """
a: ───Foo───@───
            │
b: ─────────@───
    """.strip()
    assert expected_horizontal == c.to_text_diagram().strip()

    expected_vertical = """
a   b
│   │
Foo │
│   │
@───@
│   │
    """.strip()
    assert expected_vertical == c.to_text_diagram(transpose=True).strip()


def test_named_two_qubit_diagram():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    c = cirq.Circuit(
        cirq.MatrixGate(cirq.unitary(cirq.CZ), name='Foo').on(a, b),
        cirq.MatrixGate(cirq.unitary(cirq.CZ), name='Bar').on(c, a),
    )

    expected_horizontal = """
a: ───Foo[1]───Bar[2]───
      │        │
b: ───Foo[2]───┼────────
               │
c: ────────────Bar[1]───
    """.strip()
    assert expected_horizontal == c.to_text_diagram().strip()

    expected_vertical = """
a      b      c
│      │      │
Foo[1]─Foo[2] │
│      │      │
Bar[2]─┼──────Bar[1]
│      │      │
    """.strip()
    assert expected_vertical == c.to_text_diagram(transpose=True).strip()


def test_str_executes():
    assert '1' in str(cirq.MatrixGate(np.eye(2)))
    assert '0' in str(cirq.MatrixGate(np.eye(4)))


def test_one_qubit_consistent():
    u = cirq.testing.random_unitary(2)
    g = cirq.MatrixGate(u)
    cirq.testing.assert_implements_consistent_protocols(g)


def test_two_qubit_consistent():
    u = cirq.testing.random_unitary(4)
    g = cirq.MatrixGate(u)
    cirq.testing.assert_implements_consistent_protocols(g)


def test_repr():
    cirq.testing.assert_equivalent_repr(cirq.MatrixGate(cirq.testing.random_unitary(2)))
    cirq.testing.assert_equivalent_repr(cirq.MatrixGate(cirq.testing.random_unitary(4)))


def test_matrix_gate_init_validation():
    with pytest.raises(ValueError, match='square 2d numpy array'):
        _ = cirq.MatrixGate(np.ones(shape=(1, 1, 1)))
    with pytest.raises(ValueError, match='square 2d numpy array'):
        _ = cirq.MatrixGate(np.ones(shape=(2, 1)))
    with pytest.raises(ValueError, match='not a power of 2'):
        _ = cirq.MatrixGate(np.ones(shape=(0, 0)))
    with pytest.raises(ValueError, match='not a power of 2'):
        _ = cirq.MatrixGate(np.eye(3))
    with pytest.raises(ValueError, match='matrix shape for qid_shape'):
        _ = cirq.MatrixGate(np.eye(3), qid_shape=(4,))


def test_matrix_gate_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.MatrixGate(np.eye(1)))
    eq.add_equality_group(cirq.MatrixGate(-np.eye(1)))
    eq.add_equality_group(cirq.MatrixGate(np.diag([1, 1, 1, 1, 1, -1]), qid_shape=(2, 3)))
    eq.add_equality_group(cirq.MatrixGate(np.diag([1, 1, 1, 1, 1, -1]), qid_shape=(3, 2)))


def test_matrix_gate_pow():
    t = sympy.Symbol('t')
    assert cirq.pow(cirq.MatrixGate(1j * np.eye(1)), t, default=None) is None
    assert cirq.pow(cirq.MatrixGate(1j * np.eye(1)), 2) == cirq.MatrixGate(-np.eye(1))

    m = cirq.MatrixGate(np.diag([1, 1j, -1]), qid_shape=(3,))
    assert m ** 3 == cirq.MatrixGate(np.diag([1, -1j, -1]), qid_shape=(3,))


def test_phase_by():
    # Single qubit case.
    x = cirq.MatrixGate(cirq.unitary(cirq.X))
    y = cirq.phase_by(x, 0.25, 0)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(y), cirq.unitary(cirq.Y), atol=1e-8
    )

    # Two qubit case. Commutes with control.
    cx = cirq.MatrixGate(cirq.unitary(cirq.X.controlled(1)))
    cx2 = cirq.phase_by(cx, 0.25, 0)
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(cx2), cirq.unitary(cx), atol=1e-8)

    # Two qubit case. Doesn't commute with target.
    cy = cirq.phase_by(cx, 0.25, 1)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(cy), cirq.unitary(cirq.Y.controlled(1)), atol=1e-8
    )

    m = cirq.MatrixGate(np.eye(3), qid_shape=[3])
    with pytest.raises(TypeError, match='returned NotImplemented'):
        _ = cirq.phase_by(m, 0.25, 0)


def test_protocols_and_repr():
    cirq.testing.assert_implements_consistent_protocols(cirq.MatrixGate(np.diag([1, 1j, 1, -1])))
    cirq.testing.assert_implements_consistent_protocols(
        cirq.MatrixGate(np.diag([1, 1j, -1]), qid_shape=(3,))
    )


def test_matrixgate_unitary_tolerance():
    ## non-unitary matrix
    with pytest.raises(ValueError):
        _ = cirq.MatrixGate(np.array([[1, 0], [0, -0.6]]), unitary_check_atol=0.5)

    # very high atol -> check converges quickly
    _ = cirq.MatrixGate(np.array([[1, 0], [0, 1]]), unitary_check_atol=1)

    # very high rtol -> check converges quickly
    _ = cirq.MatrixGate(np.array([[1, 0], [0, -0.6]]), unitary_check_rtol=1)

    ## unitary matrix
    _ = cirq.MatrixGate(np.array([[0.707, 0.707], [-0.707, 0.707]]), unitary_check_atol=0.5)

    # very low atol -> the check never converges
    with pytest.raises(ValueError):
        _ = cirq.MatrixGate(np.array([[0.707, 0.707], [-0.707, 0.707]]), unitary_check_atol=1e-10)

    # very low atol -> the check never converges
    with pytest.raises(ValueError):
        _ = cirq.MatrixGate(np.array([[0.707, 0.707], [-0.707, 0.707]]), unitary_check_rtol=1e-10)
