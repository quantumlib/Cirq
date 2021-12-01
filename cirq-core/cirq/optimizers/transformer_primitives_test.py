# Copyright 2021 The Cirq Developers
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
from cirq.optimizers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG


def test_map_operations_can_write_new_gates_inline():
    x = cirq.NamedQubit('x')
    y = cirq.NamedQubit('y')
    z = cirq.NamedQubit('z')
    c = cirq.Circuit(
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
    cirq.testing.assert_has_diagram(
        c,
        '''
x: ───@───Y───Z───Z───────────
      │
y: ───@───X───@───Z───@───X───
              │       │   │
z: ───────────X───────X───@───
''',
    )
    expected_diagram = '''
x: ───X───X───X───X───────────

y: ───X───X───X───X───X───X───

z: ───────────X───────X───X───
'''
    cirq.testing.assert_has_diagram(
        cirq.map_operations(c, lambda op, _: cirq.X.on_each(*op.qubits)), expected_diagram
    )
    cirq.testing.assert_has_diagram(
        cirq.map_operations_and_unroll(c, lambda op, _: cirq.X.on_each(*op.qubits)),
        expected_diagram,
    )


def test_map_operations_does_not_insert_too_many_moments():
    q = cirq.LineQubit.range(5)
    c_orig = cirq.Circuit(
        cirq.CX(q[0], q[1]),
        cirq.CX(q[3], q[2]),
        cirq.CX(q[3], q[4]),
    )

    def map_func(op: cirq.Operation, _: int) -> cirq.OP_TREE:
        if op.gate == cirq.CX:
            yield cirq.Z.on_each(*op.qubits)
            yield cirq.CX(*op.qubits)
            yield cirq.Z.on_each(*op.qubits)
        return op

    cirq.testing.assert_has_diagram(
        c_orig,
        '''
0: ───@───────
      │
1: ───X───────

2: ───X───────
      │
3: ───@───@───
          │
4: ───────X───
''',
    )

    c_mapped = cirq.map_operations(c_orig, map_func)
    circuit_op = cirq.CircuitOperation(
        cirq.FrozenCircuit(
            cirq.Z.on_each(q[0], q[1]), cirq.CNOT(q[0], q[1]), cirq.Z.on_each(q[0], q[1])
        )
    )
    c_expected = cirq.Circuit(
        circuit_op.with_qubits(q[0], q[1]).mapped_op().with_tags('<mapped_circuit_op>'),
        circuit_op.with_qubits(q[3], q[2]).mapped_op().with_tags('<mapped_circuit_op>'),
        circuit_op.with_qubits(q[3], q[4]).mapped_op().with_tags('<mapped_circuit_op>'),
    )
    cirq.testing.assert_same_circuits(c_mapped, c_expected)

    cirq.testing.assert_has_diagram(
        cirq.map_operations_and_unroll(c_orig, map_func),
        '''
0: ───Z───@───Z───────────────
          │
1: ───Z───X───Z───────────────

2: ───Z───X───Z───────────────
          │
3: ───Z───@───Z───Z───@───Z───
                      │
4: ───────────────Z───X───Z───
''',
    )


def test_unroll_circuit_op_and_variants():
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.X(q[0]), cirq.CNOT(q[0], q[1]), cirq.X(q[0]))
    cirq.testing.assert_has_diagram(
        c,
        '''
0: ───X───@───X───
          │
1: ───────X───────
''',
    )
    mapped_circuit = cirq.map_operations(
        c, lambda op, i: [cirq.Z(q[1])] * 2 if op.gate == cirq.CNOT else op
    )
    cirq.testing.assert_has_diagram(
        cirq.unroll_circuit_op(mapped_circuit),
        '''
0: ───X───────────X───

1: ───────Z───Z───────
''',
    )
    cirq.testing.assert_has_diagram(
        cirq.unroll_circuit_op_greedy_earliest(mapped_circuit),
        '''
0: ───X───────X───

1: ───Z───Z───────
''',
    )
    cirq.testing.assert_has_diagram(
        cirq.unroll_circuit_op_greedy_frontier(mapped_circuit),
        '''
0: ───X───────X───

1: ───────Z───Z───
''',
    )


def test_unroll_circuit_op_no_tags():
    q = cirq.LineQubit.range(2)
    op_list = [cirq.X(q[0]), cirq.Y(q[1])]
    op1 = cirq.CircuitOperation(cirq.FrozenCircuit(op_list))
    op2 = op1.with_tags("custom tag")
    op3 = op1.with_tags(MAPPED_CIRCUIT_OP_TAG)
    c = cirq.Circuit(op1, op2, op3)
    for unroller in [
        cirq.unroll_circuit_op,
        cirq.unroll_circuit_op_greedy_earliest,
        cirq.unroll_circuit_op_greedy_frontier,
    ]:
        cirq.testing.assert_same_circuits(
            unroller(c, tags_to_check=None), cirq.Circuit([op_list] * 3)
        )
        cirq.testing.assert_same_circuits(unroller(c), cirq.Circuit([op1, op2, op_list]))
        cirq.testing.assert_same_circuits(
            unroller(c, tags_to_check=("custom tag",)), cirq.Circuit([op1, op_list, op3])
        )
        cirq.testing.assert_same_circuits(
            unroller(
                c,
                tags_to_check=("custom tag", MAPPED_CIRCUIT_OP_TAG),
            ),
            cirq.Circuit([op1, op_list, op_list]),
        )


def test_map_operations_raises_qubits_not_subset():
    q = cirq.LineQubit.range(3)
    with pytest.raises(ValueError, match='should act on a subset'):
        _ = cirq.map_operations(
            cirq.Circuit(cirq.CNOT(q[0], q[1])), lambda op, i: cirq.CNOT(q[1], q[2])
        )


def test_map_moments_drop_empty_moments():
    op = cirq.X(cirq.NamedQubit("x"))
    c = cirq.Circuit(cirq.Moment(op), cirq.Moment(), cirq.Moment(op))
    c_mapped = cirq.map_moments(c, lambda m, i: [] if len(m) == 0 else [m])
    cirq.testing.assert_same_circuits(c_mapped, cirq.Circuit(c[0], c[0]))
