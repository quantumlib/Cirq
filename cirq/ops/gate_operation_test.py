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

import numpy as np
import pytest
import sympy
import cirq


def test_gate_operation_init():
    q = cirq.NamedQubit('q')
    g = cirq.SingleQubitGate()
    v = cirq.GateOperation(g, (q,))
    assert v.gate == g
    assert v.qubits == (q,)


def test_invalid_gate_operation():
    three_qubit_gate = cirq.ThreeQubitGate()
    single_qubit = [cirq.GridQubit(0, 0)]
    with pytest.raises(ValueError, match="number of qubits"):
        cirq.GateOperation(three_qubit_gate, single_qubit)


def test_gate_operation_eq():
    g1 = cirq.SingleQubitGate()
    g2 = cirq.SingleQubitGate()
    g3 = cirq.TwoQubitGate()
    r1 = [cirq.NamedQubit('r1')]
    r2 = [cirq.NamedQubit('r2')]
    r12 = r1 + r2
    r21 = r2 + r1

    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.GateOperation(g1, r1))
    eq.make_equality_group(lambda: cirq.GateOperation(g2, r1))
    eq.make_equality_group(lambda: cirq.GateOperation(g1, r2))
    eq.make_equality_group(lambda: cirq.GateOperation(g3, r12))
    eq.make_equality_group(lambda: cirq.GateOperation(g3, r21))
    eq.add_equality_group(cirq.GateOperation(cirq.CZ, r21),
                          cirq.GateOperation(cirq.CZ, r12))

    @cirq.value_equality
    class PairGate(cirq.Gate, cirq.InterchangeableQubitsGate):
        """Interchangeable substes."""

        def __init__(self, num_qubits):
            self._num_qubits = num_qubits

        def num_qubits(self) -> int:
            return self._num_qubits

        def qubit_index_to_equivalence_group_key(self, index: int):
            return index // 2

        def _value_equality_values_(self):
            return self.num_qubits(),

    def p(*q):
        return PairGate(len(q)).on(*q)
    a0, a1, b0, b1, c0 = cirq.LineQubit.range(5)
    eq.add_equality_group(p(a0, a1, b0, b1), p(a1, a0, b1, b0))
    eq.add_equality_group(p(b0, b1, a0, a1))
    eq.add_equality_group(p(a0, a1, b0, b1, c0), p(a1, a0, b1, b0, c0))
    eq.add_equality_group(p(a0, b0, a1, b1, c0))
    eq.add_equality_group(p(a0, c0, b0, b1, a1))
    eq.add_equality_group(p(b0, a1, a0, b1, c0))


def test_gate_operation_approx_eq():
    a = [cirq.NamedQubit('r1')]
    b = [cirq.NamedQubit('r2')]

    assert cirq.approx_eq(cirq.GateOperation(cirq.XPowGate(), a),
                          cirq.GateOperation(cirq.XPowGate(), a))
    assert not cirq.approx_eq(cirq.GateOperation(cirq.XPowGate(), a),
                              cirq.GateOperation(cirq.XPowGate(), b))

    assert cirq.approx_eq(cirq.GateOperation(cirq.XPowGate(exponent=0), a),
                          cirq.GateOperation(cirq.XPowGate(exponent=1e-9), a))
    assert not cirq.approx_eq(cirq.GateOperation(cirq.XPowGate(exponent=0), a),
                              cirq.GateOperation(cirq.XPowGate(exponent=1e-7),
                                                 a))
    assert cirq.approx_eq(cirq.GateOperation(cirq.XPowGate(exponent=0), a),
                          cirq.GateOperation(cirq.XPowGate(exponent=1e-7), a),
                          atol=1e-6)


def test_gate_operation_pow():
    Y = cirq.Y
    q = cirq.NamedQubit('q')
    assert (Y ** 0.5)(q) == Y(q) ** 0.5


def test_with_qubits_and_transform_qubits():
    g = cirq.ThreeQubitGate()
    op = cirq.GateOperation(g, cirq.LineQubit.range(3))
    assert op.with_qubits(*cirq.LineQubit.range(3, 0, -1)) \
           == cirq.GateOperation(g, cirq.LineQubit.range(3, 0, -1))
    assert op.transform_qubits(lambda e: cirq.LineQubit(-e.x)
                               ) == cirq.GateOperation(g, [cirq.LineQubit(0),
                                                           cirq.LineQubit(-1),
                                                           cirq.LineQubit(-2)])

    # The gate's constraints should be applied when changing the qubits.
    with pytest.raises(ValueError):
        _ = cirq.H(cirq.LineQubit(0)).with_qubits(cirq.LineQubit(0),
                                                  cirq.LineQubit(1))


def test_extrapolate():
    q = cirq.NamedQubit('q')

    # If the gate isn't extrapolatable, you get a type error.
    op0 = cirq.GateOperation(cirq.SingleQubitGate(), [q])
    with pytest.raises(TypeError):
        _ = op0**0.5

    op1 = cirq.GateOperation(cirq.Y, [q])
    assert op1**0.5 == cirq.GateOperation(cirq.Y**0.5, [q])
    assert (cirq.Y**0.5).on(q) == cirq.Y(q)**0.5


def test_inverse():
    q = cirq.NamedQubit('q')

    # If the gate isn't reversible, you get a type error.
    op0 = cirq.GateOperation(cirq.SingleQubitGate(), [q])
    assert cirq.inverse(op0, None) is None

    op1 = cirq.GateOperation(cirq.S, [q])
    assert cirq.inverse(op1) == op1**-1 == cirq.GateOperation(cirq.S**-1, [q])
    assert cirq.inverse(cirq.S).on(q) == cirq.inverse(cirq.S.on(q))


def test_text_diagrammable():
    q = cirq.NamedQubit('q')

    # If the gate isn't diagrammable, you get a type error.
    op0 = cirq.GateOperation(cirq.SingleQubitGate(), [q])
    with pytest.raises(TypeError):
        _ = cirq.circuit_diagram_info(op0)

    op1 = cirq.GateOperation(cirq.S, [q])
    actual = cirq.circuit_diagram_info(op1)
    expected = cirq.circuit_diagram_info(cirq.S)
    assert actual == expected


def test_bounded_effect():
    q = cirq.NamedQubit('q')

    # If the gate isn't bounded, you get a type error.
    op0 = cirq.GateOperation(cirq.SingleQubitGate(), [q])
    assert cirq.trace_distance_bound(op0) >= 1
    op1 = cirq.GateOperation(cirq.Z**0.000001, [q])
    op1_bound = cirq.trace_distance_bound(op1)
    assert op1_bound == cirq.trace_distance_bound(cirq.Z**0.000001)


def test_parameterizable_effect():
    q = cirq.NamedQubit('q')
    r = cirq.ParamResolver({'a': 0.5})

    op1 = cirq.GateOperation(cirq.Z**sympy.Symbol('a'), [q])
    assert cirq.is_parameterized(op1)
    op2 = cirq.resolve_parameters(op1, r)
    assert not cirq.is_parameterized(op2)
    assert op2 == cirq.S.on(q)


def test_pauli_expansion():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert cirq.pauli_expansion(cirq.X(a)) == cirq.LinearDict({'X': 1})
    assert (cirq.pauli_expansion(cirq.CNOT(a, b)) == cirq.pauli_expansion(
        cirq.CNOT))


def test_unitary():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert not cirq.has_unitary(cirq.measure(a))
    assert cirq.unitary(cirq.measure(a), None) is None
    np.testing.assert_allclose(cirq.unitary(cirq.X(a)),
                               np.array([[0, 1], [1, 0]]),
                               atol=1e-8)
    np.testing.assert_allclose(cirq.unitary(cirq.CNOT(a, b)),
                               cirq.unitary(cirq.CNOT),
                               atol=1e-8)


def test_channel():
    a = cirq.NamedQubit('a')
    op = cirq.bit_flip(0.5).on(a)
    np.testing.assert_allclose(cirq.channel(op), cirq.channel(op.gate))
    assert cirq.has_channel(op)

    assert cirq.channel(cirq.SingleQubitGate()(a), None) is None
    assert not cirq.has_channel(cirq.SingleQubitGate()(a))


def test_measurement_key():
    a = cirq.NamedQubit('a')
    assert cirq.measurement_key(cirq.measure(a, key='lock')) == 'lock'


def assert_mixtures_equal(actual, expected):
    """Assert equal for tuple of mixed scalar and array types."""
    for a, e in zip(actual, expected):
        np.testing.assert_almost_equal(a[0], e[0])
        np.testing.assert_almost_equal(a[1], e[1])


def test_mixture():
    a = cirq.NamedQubit('a')
    op = cirq.bit_flip(0.5).on(a)
    assert_mixtures_equal(cirq.mixture(op), cirq.mixture(op.gate))
    assert cirq.has_mixture(op)

    assert cirq.mixture(cirq.X(a), None) is None
    assert not cirq.has_mixture(cirq.X(a))


def test_repr():
    a, b = cirq.LineQubit.range(2)
    assert repr(cirq.GateOperation(cirq.CZ, (a, b))
                ) == 'cirq.CZ.on(cirq.LineQubit(0), cirq.LineQubit(1))'

    class Inconsistent(cirq.SingleQubitGate):
        def __repr__(self):
            return 'Inconsistent'

        def on(self, *qubits):
            return cirq.GateOperation(Inconsistent(), qubits)

    assert (repr(cirq.GateOperation(Inconsistent(), [a])) ==
            'cirq.GateOperation(gate=Inconsistent, qubits=[cirq.LineQubit(0)])')


def test_op_gate_of_type():
    a = cirq.NamedQubit('a')
    op = cirq.X(a)
    assert cirq.op_gate_of_type(op, cirq.XPowGate) == op.gate
    assert cirq.op_gate_of_type(op, cirq.YPowGate) is None

    class NonGateOperation(cirq.Operation):
        def qubits(self) :
            pass

        def with_qubits(self, *new_qubits):
            pass

    assert cirq.op_gate_of_type(NonGateOperation(), cirq.X) is None
