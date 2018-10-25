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

import cirq


def test_gate_operation_init():
    q = cirq.NamedQubit('q')
    g = cirq.Gate()
    v = cirq.GateOperation(g, (q,))
    assert v.gate == g
    assert v.qubits == (q,)


def test_gate_operation_eq():
    g1 = cirq.Gate()
    g2 = cirq.Gate()
    r1 = [cirq.NamedQubit('r1')]
    r2 = [cirq.NamedQubit('r2')]
    r12 = r1 + r2
    r21 = r2 + r1

    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.GateOperation(g1, r1))
    eq.make_equality_group(lambda: cirq.GateOperation(g2, r1))
    eq.make_equality_group(lambda: cirq.GateOperation(g1, r2))
    eq.make_equality_group(lambda: cirq.GateOperation(g1, r12))
    eq.make_equality_group(lambda: cirq.GateOperation(g1, r21))
    eq.add_equality_group(cirq.GateOperation(cirq.CZ, r21),
                          cirq.GateOperation(cirq.CZ, r12))

    # Interchangeable subsets.

    class PairGate(cirq.Gate, cirq.InterchangeableQubitsGate):
        def qubit_index_to_equivalence_group_key(self, index: int):
            return index // 2

    p = PairGate()
    a0, a1, b0, b1, c0 = cirq.LineQubit.range(5)
    eq.add_equality_group(p(a0, a1, b0, b1), p(a1, a0, b1, b0))
    eq.add_equality_group(p(b0, b1, a0, a1))
    eq.add_equality_group(p(a0, a1, b0, b1, c0), p(a1, a0, b1, b0, c0))
    eq.add_equality_group(p(a0, b0, a1, b1, c0))
    eq.add_equality_group(p(a0, c0, b0, b1, a1))
    eq.add_equality_group(p(b0, a1, a0, b1, c0))


def test_gate_operation_pow():
    Y = cirq.Y
    q = cirq.NamedQubit('q')
    assert (Y ** 0.5)(q) == Y(q) ** 0.5


def test_with_qubits_and_transform_qubits():
    g = cirq.Gate()
    op = cirq.GateOperation(g, cirq.LineQubit.range(3))
    assert op.with_qubits(*cirq.LineQubit.range(2)
                          ) == cirq.GateOperation(g, cirq.LineQubit.range(2))
    assert op.transform_qubits(lambda e: cirq.LineQubit(-e.x)
                               ) == cirq.GateOperation(g, [cirq.LineQubit(0),
                                                           cirq.LineQubit(-1),
                                                           cirq.LineQubit(-2)])

    # The gate's constraints should be applied when changing the qubits.
    with pytest.raises(ValueError):
        _ = cirq.Y(cirq.LineQubit(0)).with_qubits(cirq.LineQubit(0),
                                                  cirq.LineQubit(1))


def test_extrapolate():
    q = cirq.NamedQubit('q')

    # If the gate isn't extrapolatable, you get a type error.
    op0 = cirq.GateOperation(cirq.Gate(), [q])
    with pytest.raises(TypeError):
        _ = op0**0.5

    op1 = cirq.GateOperation(cirq.Y, [q])
    assert op1**0.5 == cirq.GateOperation(cirq.Y**0.5, [q])
    assert (cirq.Y**0.5).on(q) == cirq.Y(q)**0.5


def test_inverse():
    q = cirq.NamedQubit('q')

    # If the gate isn't reversible, you get a type error.
    op0 = cirq.GateOperation(cirq.Gate(), [q])
    assert cirq.inverse(op0, None) is None

    op1 = cirq.GateOperation(cirq.S, [q])
    assert cirq.inverse(op1) == op1**-1 == cirq.GateOperation(cirq.S**-1, [q])
    assert cirq.inverse(cirq.S).on(q) == cirq.inverse(cirq.S.on(q))


def test_text_diagrammable():
    q = cirq.NamedQubit('q')

    # If the gate isn't diagrammable, you get a type error.
    op0 = cirq.GateOperation(cirq.Gate(), [q])
    with pytest.raises(TypeError):
        _ = cirq.circuit_diagram_info(op0)

    op1 = cirq.GateOperation(cirq.S, [q])
    actual = cirq.circuit_diagram_info(op1)
    expected = cirq.circuit_diagram_info(cirq.S)
    assert actual == expected


def test_bounded_effect():
    q = cirq.NamedQubit('q')

    # If the gate isn't bounded, you get a type error.
    op0 = cirq.GateOperation(cirq.Gate(), [q])
    assert cirq.trace_distance_bound(op0) >= 1
    op1 = cirq.GateOperation(cirq.Z**0.000001, [q])
    op1_bound = cirq.trace_distance_bound(op1)
    assert op1_bound == cirq.trace_distance_bound(cirq.Z**0.000001)


def test_parameterizable_effect():
    q = cirq.NamedQubit('q')
    r = cirq.ParamResolver({'a': 0.5})

    op1 = cirq.GateOperation(cirq.Z**cirq.Symbol('a'), [q])
    assert cirq.is_parameterized(op1)
    op2 = cirq.resolve_parameters(op1, r)
    assert not cirq.is_parameterized(op2)
    assert op2 == cirq.S.on(q)


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


def test_repr():
    a, b = cirq.LineQubit.range(2)
    assert repr(cirq.GateOperation(cirq.CZ, (a, b))
                ) == 'cirq.CZ.on(cirq.LineQubit(0), cirq.LineQubit(1))'

    class Inconsistent(cirq.Gate):
        def __repr__(self):
            return 'Inconsistent'

        def on(self, *qubits):
            return cirq.GateOperation(Inconsistent(), qubits)

    assert (repr(cirq.GateOperation(Inconsistent(), [a])) ==
            'cirq.GateOperation(gate=Inconsistent, qubits=[cirq.LineQubit(0)])')
