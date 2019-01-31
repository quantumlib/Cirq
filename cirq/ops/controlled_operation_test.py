# Copyright 2019 The Cirq Developers
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

import cirq
import numpy as np
from cirq import protocols


def test_controlled_gate_operation_init():
    cb = cirq.NamedQubit('ctr')
    q = cirq.NamedQubit('q')
    g = cirq.Gate()
    v = cirq.GateOperation(g, (q,))
    c = cirq.ControlledOperation(cb, v)
    assert c.sub_operation == v
    assert c.control == cb
    assert c.qubits == (cb, q)

    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(c, c.with_qubits(cb, q))


def test_gate_operation_eq():
    c1 = cirq.NamedQubit('c1')
    q1 = cirq.NamedQubit('q1')
    c2 = cirq.NamedQubit('c2')

    eq = cirq.testing.EqualsTester()

    eq.make_equality_group(lambda: cirq.ControlledOperation(c1, cirq.X(q1)))
    eq.make_equality_group(lambda: cirq.ControlledOperation(c2, cirq.X(q1)))
    eq.make_equality_group(lambda: cirq.ControlledOperation(c1, cirq.Z(q1)))
    eq.make_equality_group(lambda: cirq.ControlledOperation(c2, cirq.Z(q1)))


def test_str():
    c1 = cirq.NamedQubit('c1')
    c2 = cirq.NamedQubit('c2')
    q2 = cirq.NamedQubit('q2')

    assert (str(cirq.ControlledOperation(c1, cirq.CZ(c2, q2))) ==
            "C(c1)CZ(c2, q2)")


def test_repr():
    c1 = cirq.NamedQubit('c1')
    c2 = cirq.NamedQubit('c2')
    q2 = cirq.NamedQubit('q2')

    assert (repr(cirq.ControlledOperation(c1, cirq.CZ(c2, q2))) ==
            "cirq.ControlledOperation(control=cirq.NamedQubit('c1')"
            ", sub_operation=cirq.CZ.on(cirq.NamedQubit('c2')"
            ", cirq.NamedQubit('q2')))")


# A contrived multiqubit Hadamard gate that asserts the consistency of
# the passed in Args and puts an H on all qubits
# displays them as 'H(qubit)' on the wire
class MultiH(cirq.Gate):

    def _circuit_diagram_info_(self,
                               args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        assert args.known_qubit_count is not None
        assert args.known_qubits is not None

        return protocols.CircuitDiagramInfo(
            wire_symbols=tuple('H({})'.format(q) for q in args.known_qubits),
            connected=True
        )


def test_circuit_diagram():
    qubits = cirq.LineQubit.range(3)
    c = cirq.Circuit()
    c.append(cirq.ControlledOperation(qubits[0], MultiH()(*qubits[1:])))

    cirq.testing.assert_has_diagram(c, """
0: ───@──────
      │
1: ───H(1)───
      │
2: ───H(2)───
""", use_unicode_characters=True)


class MockGate(cirq.Gate):
    def __init__(self, diagram_info: protocols.CircuitDiagramInfo):
        self.diagram_info = diagram_info

    def _circuit_diagram_info_(self,
                               args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        self.captured_diagram_args = args
        return self.diagram_info


def test_circuit_diagram_info():
    qbits = cirq.LineQubit.range(3)
    mock_gate = MockGate(None)
    c_op = cirq.ControlledOperation(qbits[0],
                                    mock_gate(*qbits[1:]))
    args = protocols.CircuitDiagramInfoArgs(
        known_qubits=None,
        known_qubit_count=None,
        use_unicode_characters=True,
        precision=1,
        qubit_map=None
    )

    assert cirq.circuit_diagram_info(c_op, args) is None
    assert mock_gate.captured_diagram_args == args


@pytest.mark.parametrize('gate', [
    cirq.X(cirq.NamedQubit('q1')),
    cirq.X(cirq.NamedQubit('q1')) ** 0.5,
    cirq.Rx(np.pi)(cirq.NamedQubit('q1')),
    cirq.Rx(np.pi / 2)(cirq.NamedQubit('q1')),
    cirq.Z(cirq.NamedQubit('q1')),
    cirq.H(cirq.NamedQubit('q1')),
    cirq.CNOT(cirq.NamedQubit('q1'), cirq.NamedQubit('q2')),
    cirq.SWAP(cirq.NamedQubit('q1'), cirq.NamedQubit('q2')),
    cirq.CCZ(cirq.NamedQubit('q1'), cirq.NamedQubit('q2'),
             cirq.NamedQubit('q3')),
    cirq.ControlledGate(cirq.ControlledGate(cirq.CCZ))(*cirq.LineQubit.range(5))
])
def test_controlled_gate_is_consistent(gate: cirq.GateOperation):
    cb = cirq.NamedQubit('ctr')
    cgate = cirq.ControlledOperation(cb, gate)
    cirq.testing.assert_implements_consistent_protocols(cgate)


def test_parameterizable():
    a = cirq.Symbol('a')
    qubits = cirq.LineQubit.range(3)

    cz = cirq.ControlledOperation(qubits[0], cirq.Z(qubits[1]))
    cza = cirq.ControlledOperation(qubits[0],
                                   cirq.ZPowGate(exponent=a)(qubits[1]))
    assert cirq.is_parameterized(cza)
    assert not cirq.is_parameterized(cz)
    assert cirq.resolve_parameters(cza, cirq.ParamResolver({'a': 1})) == cz


def test_bounded_effect():
    qubits = cirq.LineQubit.range(2)
    cy = cirq.ControlledOperation(qubits[0], cirq.Y(qubits[1]))
    assert cirq.trace_distance_bound(cy ** 0.001) < 0.01
