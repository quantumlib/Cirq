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

import cirq
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


def test_circuit_diagram_info():
    qbits = cirq.LineQubit.range(3)
    c = cirq.Circuit()
    c.append(cirq.ControlledOperation(qbits[0], MultiH()(*qbits[1:])))

    cirq.testing.assert_has_diagram(c, """
0: ───@──────
      │
1: ───H(1)───
      │
2: ───H(2)───
""", use_unicode_characters=True)

# TODO(balintp): more tests to cover
#  - None case for known qubits in diagram info
#  - parameters
#  - trace_distance
#  - __pow__
