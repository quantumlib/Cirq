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

from typing import Optional, List
import pytest

import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG


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
        yield cirq.Z.on_each(*op.qubits)
        yield cirq.CX(*op.qubits)
        yield cirq.Z.on_each(*op.qubits)

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


def test_map_operations_deep_subcircuits():
    q = cirq.LineQubit.range(5)
    c_orig = cirq.Circuit(
        cirq.CX(q[0], q[1]),
        cirq.CX(q[3], q[2]),
        cirq.CX(q[3], q[4]),
    )
    c_orig_with_circuit_ops = cirq.Circuit(
        cirq.CircuitOperation(
            cirq.FrozenCircuit(
                [cirq.CircuitOperation(cirq.FrozenCircuit(op)) for op in c_orig.all_operations()]
            )
        )
    )

    def map_func(op: cirq.Operation, _: int) -> cirq.OP_TREE:
        yield [
            cirq.Z.on_each(*op.qubits),
            cirq.CX(*op.qubits),
            cirq.Z.on_each(*op.qubits),
        ] if op.gate == cirq.CX else op

    c_mapped = cirq.map_operations(c_orig_with_circuit_ops, map_func, deep=True)
    c_mapped = cirq.unroll_circuit_op(c_mapped, deep=True, tags_to_check=None)
    cirq.testing.assert_has_diagram(
        c_mapped,
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


def test_map_operations_respects_tags_to_ignore():
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.CNOT(*q), cirq.CNOT(*q).with_tags("ignore"), cirq.CNOT(*q))
    cirq.testing.assert_same_circuits(
        cirq.Circuit(cirq.Z.on_each(*q), cirq.CNOT(*q).with_tags("ignore"), cirq.Z.on_each(*q)),
        cirq.map_operations(c, lambda op, i: cirq.Z.on_each(*op.qubits), tags_to_ignore=["ignore"]),
    )


def test_apply_tag_to_inverted_op_set():
    q = cirq.LineQubit.range(2)
    op = cirq.CNOT(*q)
    tag = "tag_to_flip"
    c_orig = cirq.Circuit(op, op.with_tags(tag), cirq.CircuitOperation(cirq.FrozenCircuit(op)))
    # Toggle with deep = True.
    c_toggled = cirq.Circuit(
        op.with_tags(tag), op, cirq.CircuitOperation(cirq.FrozenCircuit(op.with_tags(tag)))
    )
    cirq.testing.assert_same_circuits(cirq.toggle_tags(c_orig, [tag], deep=True), c_toggled)
    cirq.testing.assert_same_circuits(cirq.toggle_tags(c_toggled, [tag], deep=True), c_orig)

    # Toggle with deep = False
    c_toggled = cirq.Circuit(
        op.with_tags(tag), op, cirq.CircuitOperation(cirq.FrozenCircuit(op)).with_tags(tag)
    )
    cirq.testing.assert_same_circuits(cirq.toggle_tags(c_orig, [tag], deep=False), c_toggled)
    cirq.testing.assert_same_circuits(cirq.toggle_tags(c_toggled, [tag], deep=False), c_orig)


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
    mapped_circuit_deep = cirq.Circuit(
        [cirq.Moment(cirq.CircuitOperation(cirq.FrozenCircuit(m))) for m in mapped_circuit[:-1]],
        mapped_circuit[-1],
    )
    for unroller in [
        cirq.unroll_circuit_op_greedy_earliest,
        cirq.unroll_circuit_op_greedy_frontier,
        cirq.unroll_circuit_op,
    ]:
        cirq.testing.assert_same_circuits(
            unroller(mapped_circuit), unroller(mapped_circuit_deep, tags_to_check=None, deep=True)
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


def test_unroll_circuit_op_deep():
    q0, q1, q2 = cirq.LineQubit.range(3)
    c = cirq.Circuit(
        cirq.X(q0),
        cirq.CircuitOperation(
            cirq.FrozenCircuit(cirq.X(q1), cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q2))))
        ),
    )
    expected = cirq.Circuit(cirq.X.on_each(q0, q1, q2))
    cirq.testing.assert_same_circuits(
        cirq.unroll_circuit_op(c, tags_to_check=None, deep=True), expected
    )
    expected = cirq.Circuit(
        cirq.X.on_each(q0, q1), cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q2)))
    )
    cirq.testing.assert_same_circuits(
        cirq.unroll_circuit_op(c, tags_to_check=None, deep=False), expected
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


def test_map_operations_can_add_qubits_if_flag_false():
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.H(q[0]))
    c_mapped = cirq.map_operations(c, lambda *_: cirq.CNOT(q[0], q[1]), raise_if_add_qubits=False)
    cirq.testing.assert_same_circuits(c_mapped, cirq.Circuit(cirq.CNOT(q[0], q[1])))


def test_map_operations_can_drop_operations():
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.X(q[0]), cirq.Y(q[1]), cirq.X(q[1]), cirq.Y(q[0]))
    c_mapped = cirq.map_operations(c, lambda op, _: op if op.gate == cirq.X else [])
    c_expected = cirq.Circuit(cirq.Moment(cirq.X(q[0])), cirq.Moment(cirq.X(q[1])))
    cirq.testing.assert_same_circuits(c_mapped, c_expected)


def test_map_moments_drop_empty_moments():
    op = cirq.X(cirq.NamedQubit("x"))
    c = cirq.Circuit(cirq.Moment(op), cirq.Moment(), cirq.Moment(op))
    c_mapped = cirq.map_moments(c, lambda m, i: [] if len(m) == 0 else [m])
    cirq.testing.assert_same_circuits(c_mapped, cirq.Circuit(c[0], c[0]))


def test_merge_moments():
    q = cirq.LineQubit.range(3)
    c_orig = cirq.Circuit(
        cirq.Z.on_each(q[0], q[1]),
        cirq.Z.on_each(q[1], q[2]),
        cirq.Z.on_each(q[1], q[0]),
        strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
    )
    c_orig = cirq.Circuit(c_orig, cirq.CCX(*q), c_orig)
    cirq.testing.assert_has_diagram(
        c_orig,
        '''
0: ───Z───────Z───@───Z───────Z───
                  │
1: ───Z───Z───Z───@───Z───Z───Z───
                  │
2: ───────Z───────X───────Z───────
''',
    )

    def merge_func(m1: cirq.Moment, m2: cirq.Moment) -> Optional[cirq.Moment]:
        def is_z_moment(m):
            return all(op.gate == cirq.Z for op in m)

        if not (is_z_moment(m1) and is_z_moment(m2)):
            return None
        qubits = m1.qubits | m2.qubits

        def mul(op1, op2):
            return (op1 or op2) if not (op1 and op2) else cirq.decompose_once(op1 * op2)

        return cirq.Moment(mul(m1.operation_at(q), m2.operation_at(q)) for q in qubits)

    cirq.testing.assert_has_diagram(
        cirq.merge_moments(c_orig, merge_func),
        '''
0: ───────@───────
          │
1: ───Z───@───Z───
          │
2: ───Z───X───Z───
''',
    )


def test_merge_moments_empty_moment_as_intermediate_step():
    q = cirq.NamedQubit("q")
    c_orig = cirq.Circuit([cirq.X(q), cirq.Y(q), cirq.Z(q)] * 2, cirq.X(q) ** 0.5)

    def merge_func(m1: cirq.Moment, m2: cirq.Moment):
        gate = cirq.single_qubit_matrix_to_phxz(cirq.unitary(cirq.Circuit(m1, m2)), atol=1e-8)
        return cirq.Moment(gate.on(q) if gate else [])

    c_new = cirq.merge_moments(c_orig, merge_func)
    assert len(c_new) == 1
    assert isinstance(c_new[0][q].gate, cirq.PhasedXZGate)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(c_orig, c_new, atol=1e-8)


def test_merge_moments_empty_circuit():
    def fail_if_called_func(*_):
        assert False

    c = cirq.Circuit()
    assert cirq.merge_moments(c, fail_if_called_func) is c


def test_merge_operations_raises():
    q = cirq.LineQubit.range(3)
    c = cirq.Circuit(cirq.CZ(*q[:2]), cirq.X(q[0]))
    with pytest.raises(ValueError, match='must act on a subset of qubits'):
        cirq.merge_operations(c, lambda *_: cirq.X(q[2]))


def test_merge_operations_nothing_to_merge():
    def fail_if_called_func(*_):
        assert False

    # Empty Circuit.
    c = cirq.Circuit()
    assert cirq.merge_operations(c, fail_if_called_func) == c
    # Single moment
    q = cirq.LineQubit.range(3)
    c += cirq.Moment(cirq.CZ(*q[:2]))
    assert cirq.merge_operations(c, fail_if_called_func) == c
    # Multi moment with disjoint operations + global phase operation.
    c += cirq.Moment(cirq.X(q[2]), cirq.global_phase_operation(1j))
    assert cirq.merge_operations(c, fail_if_called_func) == c
    # Tagged operations to be ignored.
    c += cirq.Moment(cirq.CNOT(*q[:2]).with_tags("ignore"))
    assert cirq.merge_operations(c, fail_if_called_func, tags_to_ignore=["ignore"]) == c


def _create_circuit_to_merge():
    q = cirq.LineQubit.range(3)
    return cirq.Circuit(
        cirq.Moment(cirq.H.on_each(*q)),
        cirq.CNOT(q[0], q[2]),
        cirq.CNOT(*q[0:2]),
        cirq.H(q[0]),
        cirq.CZ(*q[:2]),
        cirq.X(q[0]),
        cirq.Y(q[1]),
        cirq.CNOT(*q[0:2]),
        cirq.CNOT(*q[1:3]),
        cirq.X(q[0]),
        cirq.Moment(cirq.X(q[0]).with_tags("ignore"), cirq.Y(q[1])),
        cirq.CNOT(*q[:2]),
        strategy=cirq.InsertStrategy.NEW,
    )


def test_merge_operations_merges_connected_component():
    c_orig = _create_circuit_to_merge()
    cirq.testing.assert_has_diagram(
        c_orig,
        '''
0: ───H───@───@───H───@───X───────@───────X───X['ignore']───@───
          │   │       │           │                         │
1: ───H───┼───X───────@───────Y───X───@───────Y─────────────X───
          │                           │
2: ───H───X───────────────────────────X─────────────────────────
''',
    )

    def merge_func(op1, op2):
        """Artificial example where a CZ will absorb any merge-able operation."""
        for op in [op1, op2]:
            if op.gate == cirq.CZ:
                return op
        return None

    c_new = cirq.merge_operations(c_orig, merge_func)
    cirq.testing.assert_has_diagram(
        c_new,
        '''
0: ───H───@───────────@───────────────────────────@───
          │           │                           │
1: ───────┼───────────@───────────────@───────Y───X───
          │                           │
2: ───H───X───────────────────────────X───────────────''',
    )


# pylint: disable=line-too-long
def test_merge_operations_to_circuit_op_merges_connected_component():
    c_orig = _create_circuit_to_merge()
    cirq.testing.assert_has_diagram(
        c_orig,
        '''
0: ───H───@───@───H───@───X───────@───────X───X['ignore']───@───
          │   │       │           │                         │
1: ───H───┼───X───────@───────Y───X───@───────Y─────────────X───
          │                           │
2: ───H───X───────────────────────────X─────────────────────────
''',
    )

    def can_merge(ops1: List['cirq.Operation'], ops2: List['cirq.Operation']) -> bool:
        """Artificial example where a CZ will absorb any merge-able operation."""
        return any(o.gate == cirq.CZ for op_list in [ops1, ops2] for o in op_list)

    c_new = cirq.merge_operations_to_circuit_op(
        c_orig, can_merge, merged_circuit_op_tag="merged", tags_to_ignore=["ignore"]
    )
    cirq.testing.assert_has_diagram(
        c_new,
        '''
                      [ 0: ───────@───H───@───X───@───X─── ]
0: ───H───@───────────[           │       │       │        ]─────────────────────────────────X['ignore']───@───
          │           [ 1: ───H───X───────@───Y───X─────── ]['merged']                                     │
          │           │                                                                                    │
1: ───────┼───────────#2─────────────────────────────────────────────────────────────@───────Y─────────────X───
          │                                                                          │
2: ───H───X──────────────────────────────────────────────────────────────────────────X─────────────────────────
''',
    )


def test_merge_2q_unitaries_to_circuit_op():
    c_orig = _create_circuit_to_merge()
    c_orig[-1] = c_orig[-1].with_operations(cirq.measure(cirq.LineQubit(2)))
    cirq.testing.assert_has_diagram(
        c_orig,
        '''
0: ───H───@───@───H───@───X───────@───────X───X['ignore']───@───
          │   │       │           │                         │
1: ───H───┼───X───────@───────Y───X───@───────Y─────────────X───
          │                           │
2: ───H───X───────────────────────────X─────────────────────M───
''',
    )

    c_new = cirq.merge_k_qubit_unitaries_to_circuit_op(
        c_orig, k=2, merged_circuit_op_tag="merged", tags_to_ignore=["ignore"]
    )
    cirq.testing.assert_has_diagram(
        cirq.drop_empty_moments(c_new),
        '''
      [ 0: ───H───@─── ]             [ 0: ───────@───H───@───X───@───X─── ]
0: ───[           │    ]─────────────[           │       │       │        ]────────────────────────────────────────────X['ignore']───@───
      [ 2: ───H───X─── ]['merged']   [ 1: ───H───X───────@───Y───X─────── ]['merged']                                                │
      │                              │                                                                                               │
      │                              │                                                  [ 1: ───@───Y─── ]                           │
1: ───┼──────────────────────────────#2─────────────────────────────────────────────────[       │        ]───────────────────────────X───
      │                                                                                 [ 2: ───X─────── ]['merged']
      │                                                                                 │
2: ───#2────────────────────────────────────────────────────────────────────────────────#2───────────────────────────────────────────M───''',
    )


# pylint: enable=line-too-long


def test_merge_operations_respects_tags_to_ignore():
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.CZ(*q),
        cirq.Moment(cirq.X(q[0]), cirq.Y(q[1]).with_tags("ignore")),
        cirq.Moment(cirq.X(q[0]).with_tags("ignore"), cirq.Y(q[1])),
        cirq.CZ(*q),
        [cirq.CNOT(*q), cirq.CNOT(*q).with_tags("ignore"), cirq.CNOT(*q)],
        cirq.CZ(*q),
    )
    c_merged = cirq.Circuit(
        cirq.Moment(cirq.CZ(*q)),
        cirq.Moment(cirq.Y(q[1]).with_tags("ignore")),
        cirq.Moment(cirq.X(q[0]).with_tags("ignore")),
        cirq.Moment(cirq.CZ(*q)),
        cirq.Moment(),
        cirq.Moment(cirq.CNOT(*q).with_tags("ignore")),
        cirq.Moment(cirq.CZ(*q)),
        cirq.Moment(),
    )

    def merge_func(op1, op2):
        """Artificial example where a CZ will absorb any merge-able operation."""
        return op1 if op1.gate == cirq.CZ else (op2 if op2.gate == cirq.CZ else None)

    cirq.testing.assert_same_circuits(
        cirq.merge_operations(c, merge_func, tags_to_ignore=["ignore"]), c_merged
    )


@pytest.mark.parametrize('qubit_order', ([0, 1], [1, 0]))
def test_merge_operations_deterministic_order(qubit_order):
    q = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(cirq.identity_each(*q), cirq.H.on_each(q[i] for i in qubit_order))
    cirq.testing.assert_has_diagram(
        c_orig,
        '''
0: ───I───H───
      │
1: ───I───H───''',
    )
    c_new = cirq.merge_operations(
        c_orig, lambda op1, op2: op2 if isinstance(op1.gate, cirq.IdentityGate) else None
    )
    cirq.testing.assert_has_diagram(
        c_new,
        '''
0: ───H───────

1: ───────H───''',
    )


@pytest.mark.parametrize("op_density", [0.1, 0.5, 0.9])
def test_merge_operations_complexity(op_density):
    prng = cirq.value.parse_random_state(11011)
    circuit = cirq.testing.random_circuit(20, 500, op_density, random_state=prng)
    for merge_func in [
        lambda _, __: None,
        lambda op1, _: op1,
        lambda _, op2: op2,
        lambda op1, op2: (op1, op2, None)[prng.choice(3)],
    ]:

        def wrapped_merge_func(op1, op2):
            wrapped_merge_func.num_function_calls += 1
            return merge_func(op1, op2)

        wrapped_merge_func.num_function_calls = 0
        _ = cirq.merge_operations(circuit, wrapped_merge_func)
        total_operations = len([*circuit.all_operations()])
        assert wrapped_merge_func.num_function_calls <= 2 * total_operations
