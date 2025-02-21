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

"""Tests for `parity_gates.py`."""

import numpy as np
import pytest
import sympy

import cirq


@pytest.mark.parametrize('eigen_gate_type', [cirq.XXPowGate, cirq.YYPowGate, cirq.ZZPowGate])
def test_eigen_gates_consistent_protocols(eigen_gate_type):
    cirq.testing.assert_eigengate_implements_consistent_protocols(eigen_gate_type)


def test_xx_init():
    assert cirq.XXPowGate(exponent=1).exponent == 1
    v = cirq.XXPowGate(exponent=0.5)
    assert v.exponent == 0.5


def test_xx_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.XX,
        cirq.XXPowGate(),
        cirq.XXPowGate(exponent=1, global_shift=0),
        cirq.XXPowGate(exponent=3, global_shift=0),
        cirq.XXPowGate(global_shift=100000),
    )
    eq.add_equality_group(cirq.XX**0.5, cirq.XX**2.5, cirq.XX**4.5)
    eq.add_equality_group(cirq.XX**0.25, cirq.XX**2.25, cirq.XX**-1.75)

    iXX = cirq.XXPowGate(global_shift=0.5)
    eq.add_equality_group(iXX**0.5, iXX**4.5)
    eq.add_equality_group(iXX**2.5, iXX**6.5)


def test_xx_pow():
    assert cirq.XX**0.5 != cirq.XX**-0.5
    assert cirq.XX**-1 == cirq.XX
    assert (cirq.XX**-1) ** 0.5 == cirq.XX**-0.5


def test_xx_str():
    assert str(cirq.XX) == 'XX'
    assert str(cirq.XX**0.5) == 'XX**0.5'
    assert str(cirq.XXPowGate(global_shift=0.1)) == 'XX'


def test_xx_repr():
    assert repr(cirq.XXPowGate()) == 'cirq.XX'
    assert repr(cirq.XXPowGate(exponent=0.5)) == '(cirq.XX**0.5)'


def test_xx_matrix():
    np.testing.assert_allclose(
        cirq.unitary(cirq.XX),
        np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]),
        atol=1e-8,
    )
    np.testing.assert_allclose(cirq.unitary(cirq.XX**2), np.eye(4), atol=1e-8)
    c = np.cos(np.pi / 6)
    s = -1j * np.sin(np.pi / 6)
    np.testing.assert_allclose(
        cirq.unitary(cirq.XXPowGate(exponent=1 / 3, global_shift=-0.5)),
        np.array([[c, 0, 0, s], [0, c, s, 0], [0, s, c, 0], [s, 0, 0, c]]),
        atol=1e-8,
    )


def test_xx_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    circuit = cirq.Circuit(cirq.XX(a, b), cirq.XX(a, b) ** 3, cirq.XX(a, b) ** 0.5)
    cirq.testing.assert_has_diagram(
        circuit,
        """
a: ───XX───XX───XX───────
      │    │    │
b: ───XX───XX───XX^0.5───
""",
    )


def test_yy_init():
    assert cirq.YYPowGate(exponent=1).exponent == 1
    v = cirq.YYPowGate(exponent=0.5)
    assert v.exponent == 0.5


def test_yy_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.YY,
        cirq.YYPowGate(),
        cirq.YYPowGate(exponent=1, global_shift=0),
        cirq.YYPowGate(exponent=3, global_shift=0),
    )
    eq.add_equality_group(cirq.YY**0.5, cirq.YY**2.5, cirq.YY**4.5)
    eq.add_equality_group(cirq.YY**0.25, cirq.YY**2.25, cirq.YY**-1.75)

    iYY = cirq.YYPowGate(global_shift=0.5)
    eq.add_equality_group(iYY**0.5, iYY**4.5)
    eq.add_equality_group(iYY**2.5, iYY**6.5)


def test_yy_pow():
    assert cirq.YY**0.5 != cirq.YY**-0.5
    assert cirq.YY**-1 == cirq.YY
    assert (cirq.YY**-1) ** 0.5 == cirq.YY**-0.5


def test_yy_str():
    assert str(cirq.YY) == 'YY'
    assert str(cirq.YY**0.5) == 'YY**0.5'
    assert str(cirq.YYPowGate(global_shift=0.1)) == 'YY'

    iYY = cirq.YYPowGate(global_shift=0.5)
    assert str(iYY) == 'YY'


def test_yy_repr():
    assert repr(cirq.YYPowGate()) == 'cirq.YY'
    assert repr(cirq.YYPowGate(exponent=0.5)) == '(cirq.YY**0.5)'


def test_yy_matrix():
    np.testing.assert_allclose(
        cirq.unitary(cirq.YY),
        np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]]),
        atol=1e-8,
    )
    np.testing.assert_allclose(cirq.unitary(cirq.YY**2), np.eye(4), atol=1e-8)
    c = np.cos(np.pi / 6)
    s = 1j * np.sin(np.pi / 6)
    np.testing.assert_allclose(
        cirq.unitary(cirq.YYPowGate(exponent=1 / 3, global_shift=-0.5)),
        np.array([[c, 0, 0, s], [0, c, -s, 0], [0, -s, c, 0], [s, 0, 0, c]]),
        atol=1e-8,
    )


def test_yy_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    circuit = cirq.Circuit(cirq.YY(a, b), cirq.YY(a, b) ** 3, cirq.YY(a, b) ** 0.5)
    cirq.testing.assert_has_diagram(
        circuit,
        """
a: ───YY───YY───YY───────
      │    │    │
b: ───YY───YY───YY^0.5───
""",
    )


def test_zz_init():
    assert cirq.ZZPowGate(exponent=1).exponent == 1
    v = cirq.ZZPowGate(exponent=0.5)
    assert v.exponent == 0.5


def test_zz_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.ZZ,
        cirq.ZZPowGate(),
        cirq.ZZPowGate(exponent=1, global_shift=0),
        cirq.ZZPowGate(exponent=3, global_shift=0),
    )
    eq.add_equality_group(cirq.ZZ**0.5, cirq.ZZ**2.5, cirq.ZZ**4.5)
    eq.add_equality_group(cirq.ZZ**0.25, cirq.ZZ**2.25, cirq.ZZ**-1.75)

    iZZ = cirq.ZZPowGate(global_shift=0.5)
    eq.add_equality_group(iZZ**0.5, iZZ**4.5)
    eq.add_equality_group(iZZ**2.5, iZZ**6.5)


def test_zz_pow():
    assert cirq.ZZ**0.5 != cirq.ZZ**-0.5
    assert cirq.ZZ**-1 == cirq.ZZ
    assert (cirq.ZZ**-1) ** 0.5 == cirq.ZZ**-0.5


def test_zz_phase_by():
    assert cirq.phase_by(cirq.ZZ, 0.25, 0) == cirq.phase_by(cirq.ZZ, 0.25, 1) == cirq.ZZ
    assert cirq.phase_by(cirq.ZZ**0.5, 0.25, 0) == cirq.ZZ**0.5
    assert cirq.phase_by(cirq.ZZ**-0.5, 0.25, 1) == cirq.ZZ**-0.5


def test_zz_str():
    assert str(cirq.ZZ) == 'ZZ'
    assert str(cirq.ZZ**0.5) == 'ZZ**0.5'
    assert str(cirq.ZZPowGate(global_shift=0.1)) == 'ZZ'

    iZZ = cirq.ZZPowGate(global_shift=0.5)
    assert str(iZZ) == 'ZZ'


def test_zz_repr():
    assert repr(cirq.ZZPowGate()) == 'cirq.ZZ'
    assert repr(cirq.ZZPowGate(exponent=0.5)) == '(cirq.ZZ**0.5)'


def test_zz_matrix():
    np.testing.assert_allclose(
        cirq.unitary(cirq.ZZ),
        np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
        atol=1e-8,
    )
    np.testing.assert_allclose(cirq.unitary(cirq.ZZ**2), np.eye(4), atol=1e-8)
    b = 1j**0.25
    a = np.conj(b)
    np.testing.assert_allclose(
        cirq.unitary(cirq.ZZPowGate(exponent=0.25, global_shift=-0.5)),
        np.array([[a, 0, 0, 0], [0, b, 0, 0], [0, 0, b, 0], [0, 0, 0, a]]),
        atol=1e-8,
    )


def test_zz_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    circuit = cirq.Circuit(cirq.ZZ(a, b), cirq.ZZ(a, b) ** 3, cirq.ZZ(a, b) ** 0.5)
    cirq.testing.assert_has_diagram(
        circuit,
        """
a: ───ZZ───ZZ───ZZ───────
      │    │    │
b: ───ZZ───ZZ───ZZ^0.5───
""",
    )


def test_trace_distance():
    foo = sympy.Symbol('foo')
    assert cirq.trace_distance_bound(cirq.XX**foo) == 1.0
    assert cirq.trace_distance_bound(cirq.YY**foo) == 1.0
    assert cirq.trace_distance_bound(cirq.ZZ**foo) == 1.0
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.XX), 1.0)
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.YY**0), 0)
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.ZZ ** (1 / 3)), np.sin(np.pi / 6))


def test_ms_arguments():
    eq_tester = cirq.testing.EqualsTester()
    eq_tester.add_equality_group(
        cirq.ms(np.pi / 2), cirq.MSGate(rads=np.pi / 2), cirq.XXPowGate(global_shift=-0.5)
    )
    eq_tester.add_equality_group(
        cirq.ms(np.pi / 4), cirq.XXPowGate(exponent=0.5, global_shift=-0.5)
    )
    eq_tester.add_equality_group(cirq.XX)
    eq_tester.add_equality_group(cirq.XX**0.5)


def test_ms_equal_up_to_global_phase():
    assert cirq.equal_up_to_global_phase(cirq.ms(np.pi / 2), cirq.XX)
    assert cirq.equal_up_to_global_phase(cirq.ms(np.pi / 4), cirq.XX**0.5)
    assert not cirq.equal_up_to_global_phase(cirq.ms(np.pi / 4), cirq.XX)

    assert cirq.ms(np.pi / 2) in cirq.GateFamily(cirq.XX)
    assert cirq.ms(np.pi / 4) in cirq.GateFamily(cirq.XX**0.5)
    assert cirq.ms(np.pi / 4) not in cirq.GateFamily(cirq.XX)


def test_ms_str():
    ms = cirq.ms(np.pi / 2)
    assert str(ms) == 'MS(π/2)'
    assert str(cirq.ms(np.pi)) == 'MS(2.0π/2)'
    assert str(ms**0.5) == 'MS(0.5π/2)'
    assert str(ms**2) == 'MS(2.0π/2)'
    assert str(ms**-1) == 'MS(-1.0π/2)'


def test_ms_matrix():
    s = np.sqrt(0.5)
    # yapf: disable
    np.testing.assert_allclose(cirq.unitary(cirq.ms(np.pi/4)),
                       np.array([[s, 0, 0, -1j*s],
                                 [0, s, -1j*s, 0],
                                 [0, -1j*s, s, 0],
                                 [-1j*s, 0, 0, s]]),
                                 atol=1e-8)
    # yapf: enable
    np.testing.assert_allclose(cirq.unitary(cirq.ms(np.pi)), np.diag([-1, -1, -1, -1]), atol=1e-8)


def test_ms_repr():
    assert repr(cirq.ms(np.pi / 2)) == 'cirq.ms(np.pi/2)'
    assert repr(cirq.ms(np.pi / 4)) == 'cirq.ms(0.5*np.pi/2)'
    cirq.testing.assert_equivalent_repr(cirq.ms(np.pi / 4))
    ms = cirq.ms(np.pi / 2)
    assert repr(ms**2) == 'cirq.ms(2.0*np.pi/2)'
    assert repr(ms**-0.5) == 'cirq.ms(-0.5*np.pi/2)'


def test_ms_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    circuit = cirq.Circuit(cirq.SWAP(a, b), cirq.X(a), cirq.Y(a), cirq.ms(np.pi).on(a, b))
    cirq.testing.assert_has_diagram(
        circuit,
        """
a: ───×───X───Y───MS(π)───
      │           │
b: ───×───────────MS(π)───
""",
    )


def test_json_serialization():
    assert cirq.read_json(json_text=cirq.to_json(cirq.ms(np.pi / 2))) == cirq.ms(np.pi / 2)


@pytest.mark.parametrize('gate_cls', (cirq.XXPowGate, cirq.YYPowGate, cirq.ZZPowGate))
@pytest.mark.parametrize(
    'exponent,is_clifford',
    ((0, True), (0.5, True), (0.75, False), (1, True), (1.5, True), (-1.5, True)),
)
def test_clifford_protocols(gate_cls: type[cirq.EigenGate], exponent: float, is_clifford: bool):
    gate = gate_cls(exponent=exponent)
    assert hasattr(gate, '_decompose_into_clifford_with_qubits_')
    if is_clifford:
        clifford_decomposition = cirq.Circuit(
            gate._decompose_into_clifford_with_qubits_(cirq.LineQubit.range(2))
        )
        assert cirq.has_stabilizer_effect(gate)
        assert cirq.has_stabilizer_effect(clifford_decomposition)
        if exponent == 0:
            assert clifford_decomposition == cirq.Circuit()
        else:
            np.testing.assert_allclose(cirq.unitary(gate), cirq.unitary(clifford_decomposition))
    else:
        assert not cirq.has_stabilizer_effect(gate)
        assert gate._decompose_into_clifford_with_qubits_(cirq.LineQubit.range(2)) is NotImplemented
