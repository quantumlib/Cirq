# Copyright 2018 Google LLC
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
from cirq import PointOptimizer, PointOptimizationSummary
from cirq.testing import EqualsTester


def test_equality():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    xa = cirq.X(a)
    ya = cirq.Y(a)

    eq = EqualsTester()

    eq.make_equality_pair(lambda: PointOptimizationSummary(clear_span=0,
                                                           clear_qubits=[],
                                                           new_operations=[]))
    eq.add_equality_group(PointOptimizationSummary(clear_span=1,
                                                   clear_qubits=[a],
                                                   new_operations=[]))
    eq.add_equality_group(PointOptimizationSummary(clear_span=1,
                                                   clear_qubits=[a],
                                                   new_operations=[xa]))
    eq.add_equality_group(PointOptimizationSummary(clear_span=1,
                                                   clear_qubits=[a, b],
                                                   new_operations=[xa]))
    eq.add_equality_group(PointOptimizationSummary(clear_span=2,
                                                   clear_qubits=[a],
                                                   new_operations=[xa]))
    eq.add_equality_group(PointOptimizationSummary(clear_span=1,
                                                   clear_qubits=[a],
                                                   new_operations=[ya]))
    eq.add_equality_group(PointOptimizationSummary(clear_span=1,
                                                   clear_qubits=[a],
                                                   new_operations=[xa, xa]))


def test_point_optimizer_can_write_new_gates_inline():

    class ReplaceWithXGates(PointOptimizer):
        """Replaces a block of operations with X gates.

        Searches ahead for gates covering a subset of the focused operation's
        qubits, clears the whole range, and inserts X gates for each cleared
        operation's qubits.
        """
        def optimization_at(self, circuit, index, op):
            end = index + 1
            new_ops = [cirq.X(q) for q in op.qubits]
            done = False
            while not done:
                n = circuit.next_moment_operating_on(op.qubits, end)
                if n is None:
                    break
                next_ops = {circuit.operation_at(q, n) for q in op.qubits}
                next_ops = [e for e in next_ops if e]
                next_ops = sorted(next_ops, key=lambda e: str(e.qubits))
                for next_op in next_ops:
                    if next_op:
                        if set(next_op.qubits).issubset(op.qubits):
                            end = n + 1
                            new_ops.extend(cirq.X(q) for q in next_op.qubits)
                        else:
                            done = True

            return PointOptimizationSummary(clear_span=end - index,
                                            clear_qubits=op.qubits,
                                            new_operations=new_ops)

    x = cirq.NamedQubit('x')
    y = cirq.NamedQubit('y')
    z = cirq.NamedQubit('z')
    c = cirq.Circuit.from_ops(
        cirq.CZ(x, y),
        cirq.Y(x),
        cirq.Z(x),
        cirq.X(y),
        cirq.CNOT(y, z),
        cirq.Z(y),
        cirq.Z(x),
        cirq.CNOT(y, z),
        cirq.CNOT(z, y),
    )

    ReplaceWithXGates().optimize_circuit(c)

    assert c == cirq.Circuit([
        cirq.Moment([cirq.X(x), cirq.X(y)]),
        cirq.Moment([cirq.X(x)]),
        cirq.Moment([cirq.X(x), cirq.X(y)]),
        cirq.Moment([cirq.X(y), cirq.X(z)]),
        cirq.Moment([cirq.X(y), cirq.X(x)]),
        cirq.Moment([cirq.X(y), cirq.X(z)]),
        cirq.Moment([cirq.X(z), cirq.X(y)]),
    ])
