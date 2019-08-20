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
import sympy
import cirq


def test_parallel_gate_operation_init():
    q = cirq.NamedQubit('q')
    r = cirq.NamedQubit('r')
    g = cirq.SingleQubitGate()
    v = cirq.ParallelGateOperation(g, (q, r))
    assert v.gate == g
    assert v.qubits == (q, r)


def test_invalid_parallel_gate_operation():
    three_qubit_gate = cirq.ThreeQubitGate()
    single_qubit_gate = cirq.SingleQubitGate()
    repeated_qubits = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 0)]
    with pytest.raises(ValueError) as wrong_gate:
        cirq.ParallelGateOperation(three_qubit_gate, cirq.NamedQubit("a"))
    assert str(wrong_gate.value) == "gate must be a single qubit gate"
    with pytest.raises(ValueError) as bad_qubits:
        cirq.ParallelGateOperation(single_qubit_gate, repeated_qubits)
    assert str(bad_qubits.value) == "repeated qubits are not allowed"


def test_gate_operation_eq():
    g1 = cirq.SingleQubitGate()
    g2 = cirq.SingleQubitGate()
    r1 = [cirq.NamedQubit('r1')]
    r2 = [cirq.NamedQubit('r2')]
    r12 = r1 + r2
    r21 = r2 + r1

    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.ParallelGateOperation(g1, r1))
    eq.make_equality_group(lambda: cirq.ParallelGateOperation(g2, r1))
    eq.add_equality_group(cirq.ParallelGateOperation(g1, r12),
                          cirq.ParallelGateOperation(g1, r21))


def test_with_qubits_and_transform_qubits():
    g = cirq.SingleQubitGate()
    op = cirq.ParallelGateOperation(g, cirq.LineQubit.range(3))
    line = cirq.LineQubit.range(3, 0, -1)
    negline = cirq.LineQubit.range(0, -3, -1)
    assert op.with_qubits(*line) == cirq.ParallelGateOperation(g, line)
    assert op.transform_qubits(lambda e: cirq.LineQubit(-e.x)
                               ) == cirq.ParallelGateOperation(g, negline)


def test_decompose():
    qreg = cirq.LineQubit.range(3)
    op = cirq.ParallelGateOperation(cirq.X, qreg)
    assert cirq.decompose(op) == cirq.X.on_each(*qreg)


def test_extrapolate():
    q = cirq.NamedQubit('q')
    # If the gate isn't extrapolatable, you get a type error.
    op0 = cirq.ParallelGateOperation(cirq.SingleQubitGate(), [q])
    with pytest.raises(TypeError):
        _ = op0**0.5

    op = cirq.ParallelGateOperation(cirq.Y, [q])
    assert op**0.5 == cirq.ParallelGateOperation(cirq.Y**0.5, [q])
    assert cirq.inverse(op) == op**-1 == cirq.ParallelGateOperation(cirq.Y**-1,
                                                                    [q])


def test_parameterizable_effect():
    q = cirq.NamedQubit('q')
    r = cirq.ParamResolver({'a': 0.5})

    op1 = cirq.ParallelGateOperation(cirq.Z**sympy.Symbol('a'), [q])
    assert cirq.is_parameterized(op1)
    op2 = cirq.resolve_parameters(op1, r)
    assert not cirq.is_parameterized(op2)


def test_unitary():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    g = cirq.SingleQubitGate()
    p = cirq.ParallelGateOperation(g, [a, b])
    q = cirq.ParallelGateOperation(cirq.X, [a, b])

    assert not cirq.has_unitary(p)
    assert cirq.unitary(p, None) is None
    np.testing.assert_allclose(cirq.unitary(q),
                               np.array(
                                   [[0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j],
                                    [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                                    [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                                    [1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j]]),
                               atol=1e-8)


def test_not_implemented_diagram():
    a = cirq.NamedQubit('a')
    g = cirq.SingleQubitGate()
    c = cirq.Circuit()
    c.append(cirq.ParallelGateOperation(g,[a]))
    assert 'cirq.ops.gate_features.SingleQubitGate' in str(c)


def test_repr():
    a, b = cirq.LineQubit.range(2)
    assert repr(cirq.ParallelGateOperation(cirq.X, (a, b))
                ) == 'cirq.ParallelGateOperation(gate=cirq.X,' \
                     ' qubits=[cirq.LineQubit(0), cirq.LineQubit(1)])'


def test_str():
    a, b = cirq.LineQubit.range(2)
    assert str(cirq.ParallelGateOperation(cirq.X, (a, b))) == 'X(0, 1)'


def test_equivalent_circuit():
    qreg = cirq.LineQubit.range(4)
    oldc = cirq.Circuit()
    newc = cirq.Circuit()
    gates = [cirq.XPowGate()**(1/2), cirq.YPowGate()**(1/3),
             cirq.ZPowGate()**-1]

    for gate in gates:
        for qubit in qreg:
            oldc.append(gate.on(qubit))
        newc.append(cirq.ops.ParallelGateOperation(gate, qreg))

    cirq.testing.assert_has_diagram(newc, oldc.to_text_diagram())
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(oldc,
                                                                           newc,
                                                                           atol=
                                                                           1e-6)


@pytest.mark.parametrize('gate, qubits', (
    (cirq.X, (cirq.NamedQubit('q'),)),
    (cirq.Y, cirq.LineQubit.range(2)),
    (cirq.Z, cirq.LineQubit.range(3)),
    (cirq.H, cirq.LineQubit.range(4)),
))
def test_parallel_gate_operation_is_consistent(gate, qubits):
    op = cirq.ParallelGateOperation(gate, qubits)
    cirq.testing.assert_implements_consistent_protocols(op)


def test_trace_distance():
    s = cirq.X**0.25
    twoop = cirq.ParallelGateOperation(s, cirq.LineQubit.range(2))
    threeop = cirq.ParallelGateOperation(s, cirq.LineQubit.range(3))
    fourop = cirq.ParallelGateOperation(s, cirq.LineQubit.range(4))
    assert cirq.approx_eq(cirq.trace_distance_bound(twoop), np.sin(np.pi / 4))
    assert cirq.approx_eq(cirq.trace_distance_bound(threeop),
                          np.sin(3 * np.pi / 8))
    assert cirq.approx_eq(cirq.trace_distance_bound(fourop), 1.0)
    foo = sympy.Symbol('foo')
    spo = cirq.ParallelGateOperation(cirq.X**foo, cirq.LineQubit.range(4))
    assert cirq.approx_eq(cirq.trace_distance_bound(spo), 1.0)
