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
    g1 = cirq.Gate()
    g2 = cirq.Gate()
    c1 = cirq.NamedQubit('c1')
    q1 = [cirq.NamedQubit('q1')]
    c2 = cirq.NamedQubit('c2')
    q2 = [cirq.NamedQubit('q2')]

    eq = cirq.testing.EqualsTester()

    eq.make_equality_group(lambda:
                           cirq.ControlledOperation(c1,
                                                    cirq.GateOperation(g1, q1)))
    eq.make_equality_group(lambda:
                           cirq.ControlledOperation(c2,
                                                    cirq.GateOperation(g1, q1)))
    eq.make_equality_group(lambda:
                           cirq.ControlledOperation(c2,
                                                    cirq.GateOperation(g2, q1)))
    eq.make_equality_group(lambda:
                           cirq.ControlledOperation(c1,
                                                    cirq.GateOperation(g1, q2)))

## TODO(balintp): more tests to cover magic methods
