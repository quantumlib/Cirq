# Copyright 2022 The Cirq Developers
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

# pylint: skip-file

from typing import List

import numpy as np
import pytest

import cirq


def assert_optimizes(optimized: cirq.AbstractCircuit, expected: cirq.AbstractCircuit):
    # Ignore differences that would be caught by follow-up optimizations.
    followup_transformers: List[cirq.TRANSFORMER] = [
        cirq.drop_negligible_operations,
        cirq.drop_empty_moments,
    ]
    for transform in followup_transformers:
        optimized = transform(optimized)
        expected = transform(expected)

    cirq.testing.assert_same_circuits(optimized, expected)


def test_merge_1q_unitaries():
    q, q2 = cirq.LineQubit.range(2)
    # 1. Combines trivial 1q sequence.
    c = cirq.Circuit(cirq.X(q) ** 0.5, cirq.Z(q) ** 0.5, cirq.X(q) ** -0.5)
    c = cirq.merge_k_qubit_unitaries(c, k=1)
    op_list = [*c.all_operations()]
    assert len(op_list) == 1
    assert isinstance(op_list[0].gate, cirq.MatrixGate)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(c), cirq.unitary(cirq.Y**0.5), atol=1e-7
    )

    # 2. Gets blocked at a 2q operation.
    c = cirq.Circuit([cirq.Z(q), cirq.H(q), cirq.X(q), cirq.H(q), cirq.CZ(q, q2), cirq.H(q)])
    c = cirq.drop_empty_moments(cirq.merge_k_qubit_unitaries(c, k=1))
    assert len(c) == 3
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(c[0]), np.eye(2), atol=1e-7)
    assert isinstance(c[-1][q].gate, cirq.MatrixGate)


def test_respects_nocompile_tags():
    q = cirq.NamedQubit("q")
    c = cirq.Circuit(
        [cirq.Z(q), cirq.H(q), cirq.X(q), cirq.H(q), cirq.X(q).with_tags("nocompile"), cirq.H(q)]
    )
    context = cirq.TransformerContext(tags_to_ignore=("nocompile",))
    c = cirq.drop_empty_moments(cirq.merge_k_qubit_unitaries(c, k=1, context=context))
    assert len(c) == 3
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(c[0]), np.eye(2), atol=1e-7)
    assert c[1][q] == cirq.X(q).with_tags("nocompile")
    assert isinstance(c[-1][q].gate, cirq.MatrixGate)


def test_ignores_2qubit_target():
    c = cirq.Circuit(cirq.CZ(*cirq.LineQubit.range(2)))
    assert_optimizes(optimized=cirq.merge_k_qubit_unitaries(c, k=1), expected=c)


def test_ignore_unsupported_gate():
    class UnsupportedDummy(cirq.Gate):
        def _num_qubits_(self) -> int:
            return 1

    c = cirq.Circuit(UnsupportedDummy()(cirq.LineQubit(0)))
    assert_optimizes(optimized=cirq.merge_k_qubit_unitaries(c, k=1), expected=c)


def test_1q_rewrite():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.X(q0), cirq.Y(q0), cirq.X(q1), cirq.CZ(q0, q1), cirq.Y(q1), cirq.measure(q0, q1)
    )
    assert_optimizes(
        optimized=cirq.merge_k_qubit_unitaries(
            circuit, k=1, rewriter=lambda ops: cirq.H(ops.qubits[0])
        ),
        expected=cirq.Circuit(
            cirq.H(q0), cirq.H(q1), cirq.CZ(q0, q1), cirq.H(q1), cirq.measure(q0, q1)
        ),
    )


def test_merge_k_qubit_unitaries_raises():
    with pytest.raises(ValueError, match="k should be greater than or equal to 1"):
        _ = cirq.merge_k_qubit_unitaries(cirq.Circuit())


def test_merge_complex_circuit_preserving_moment_structure():
    q = cirq.LineQubit.range(3)
    c_orig = cirq.Circuit(
        cirq.Moment(cirq.H.on_each(*q)),
        cirq.CNOT(q[0], q[2]),
        cirq.CNOT(*q[0:2]),
        cirq.H(q[0]),
        cirq.CZ(*q[:2]),
        cirq.X(q[0]),
        cirq.Y(q[1]),
        cirq.CNOT(*q[0:2]),
        cirq.CNOT(*q[1:3]).with_tags("ignore"),
        cirq.X(q[0]),
        cirq.Moment(cirq.X(q[0]).with_tags("ignore"), cirq.Y(q[1]), cirq.Z(q[2])),
        cirq.Moment(cirq.CNOT(*q[:2]), cirq.measure(q[2], key="a")),
        cirq.X(q[0]).with_classical_controls("a"),
        strategy=cirq.InsertStrategy.NEW,
    )
    cirq.testing.assert_has_diagram(
        c_orig,
        '''
0: ───H───@───@───H───@───X───────@─────────────────X───X['ignore']───@───X───
          │   │       │           │                                   │   ║
1: ───H───┼───X───────@───────Y───X───@['ignore']───────Y─────────────X───╫───
          │                           │                                   ║
2: ───H───X───────────────────────────X─────────────────Z─────────────M───╫───
                                                                      ║   ║
a: ═══════════════════════════════════════════════════════════════════@═══^═══
''',
    )
    component_id = 0

    def rewriter_merge_to_circuit_op(op: 'cirq.CircuitOperation') -> 'cirq.OP_TREE':
        nonlocal component_id
        component_id = component_id + 1
        return op.with_tags(f'{component_id}')

    c_new = cirq.merge_k_qubit_unitaries(
        c_orig,
        k=2,
        context=cirq.TransformerContext(tags_to_ignore=("ignore",)),
        rewriter=rewriter_merge_to_circuit_op,
    )
    cirq.testing.assert_has_diagram(
        cirq.drop_empty_moments(c_new),
        '''
      [ 0: ───H───@─── ]        [ 0: ───────@───H───@───X───@───X─── ]                                            [ 0: ───────@─── ]
0: ───[           │    ]────────[           │       │       │        ]──────────────────────X['ignore']───────────[           │    ]────────X───
      [ 2: ───H───X─── ]['1']   [ 1: ───H───X───────@───Y───X─────── ]['2']                                       [ 1: ───Y───X─── ]['4']   ║
      │                         │                                                                                 │                         ║
1: ───┼─────────────────────────#2────────────────────────────────────────────@['ignore']─────────────────────────#2────────────────────────╫───
      │                                                                       │                                                             ║
2: ───#2──────────────────────────────────────────────────────────────────────X─────────────[ 2: ───Z─── ]['3']───M─────────────────────────╫───
                                                                                                                  ║                         ║
a: ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════@═════════════════════════^═══''',
    )

    component_id = 0

    def rewriter_replace_with_decomp(op: 'cirq.CircuitOperation') -> 'cirq.OP_TREE':
        nonlocal component_id
        component_id = component_id + 1
        tag = f'{component_id}'
        if len(op.qubits) == 1:
            return [cirq.T(op.qubits[0]).with_tags(tag)]
        one_layer = [op.with_tags(tag) for op in cirq.T.on_each(*op.qubits)]
        two_layer = [cirq.SQRT_ISWAP(*op.qubits).with_tags(tag)]
        return [one_layer, two_layer, one_layer]

    c_new = cirq.merge_k_qubit_unitaries(
        c_orig,
        k=2,
        context=cirq.TransformerContext(tags_to_ignore=("ignore",)),
        rewriter=rewriter_replace_with_decomp,
    )
    cirq.testing.assert_has_diagram(
        cirq.drop_empty_moments(c_new),
        '''
0: ───T['1']───iSwap['1']───T['1']───T['2']───iSwap['2']───T['2']─────────────────X['ignore']───T['4']───iSwap['4']───T['4']───X───
               │                              │                                                          │                     ║
1: ────────────┼─────────────────────T['2']───iSwap^0.5────T['2']───@['ignore']─────────────────T['4']───iSwap^0.5────T['4']───╫───
               │                                                    │                                                          ║
2: ───T['1']───iSwap^0.5────T['1']──────────────────────────────────X─────────────T['3']────────M──────────────────────────────╫───
                                                                                                ║                              ║
a: ═════════════════════════════════════════════════════════════════════════════════════════════@══════════════════════════════^═══''',
    )


def test_merge_k_qubit_unitaries_deep():
    q = cirq.LineQubit.range(2)
    h_cz_y = [cirq.H(q[0]), cirq.CZ(*q), cirq.Y(q[1])]
    c_orig = cirq.Circuit(
        h_cz_y,
        cirq.Moment(cirq.X(q[0]).with_tags("ignore"), cirq.Y(q[1])),
        cirq.CircuitOperation(cirq.FrozenCircuit(h_cz_y)).repeat(6).with_tags("ignore"),
        [cirq.CNOT(*q), cirq.CNOT(*q)],
        cirq.CircuitOperation(cirq.FrozenCircuit(h_cz_y)).repeat(4),
        [cirq.CNOT(*q), cirq.CZ(*q), cirq.CNOT(*q)],
        cirq.CircuitOperation(cirq.FrozenCircuit(h_cz_y)).repeat(5).with_tags("preserve_tag"),
    )

    def _wrap_in_cop(ops: cirq.OP_TREE, tag: str):
        return cirq.CircuitOperation(cirq.FrozenCircuit(ops)).with_tags(tag)

    c_expected = cirq.Circuit(
        _wrap_in_cop([h_cz_y, cirq.Y(q[1])], '1'),
        cirq.Moment(cirq.X(q[0]).with_tags("ignore")),
        cirq.CircuitOperation(cirq.FrozenCircuit(h_cz_y)).repeat(6).with_tags("ignore"),
        _wrap_in_cop([cirq.CNOT(*q), cirq.CNOT(*q)], '2'),
        cirq.CircuitOperation(cirq.FrozenCircuit(_wrap_in_cop(h_cz_y, '3'))).repeat(4),
        _wrap_in_cop([cirq.CNOT(*q), cirq.CZ(*q), cirq.CNOT(*q)], '4'),
        cirq.CircuitOperation(cirq.FrozenCircuit(_wrap_in_cop(h_cz_y, '5')))
        .repeat(5)
        .with_tags("preserve_tag"),
        strategy=cirq.InsertStrategy.NEW,
    )

    component_id = 0

    def rewriter_merge_to_circuit_op(op: 'cirq.CircuitOperation') -> 'cirq.OP_TREE':
        nonlocal component_id
        component_id = component_id + 1
        return op.with_tags(f'{component_id}')

    context = cirq.TransformerContext(tags_to_ignore=("ignore",), deep=True)
    c_new = cirq.merge_k_qubit_unitaries(
        c_orig,
        k=2,
        context=context,
        rewriter=rewriter_merge_to_circuit_op,
    )
    cirq.testing.assert_same_circuits(c_new, c_expected)

    def _wrap_in_matrix_gate(ops: cirq.OP_TREE):
        op = _wrap_in_cop(ops, 'temp')
        return cirq.MatrixGate(cirq.unitary(op)).on(*op.qubits)

    c_expected_matrix = cirq.Circuit(
        _wrap_in_matrix_gate([h_cz_y, cirq.Y(q[1])]),
        cirq.Moment(cirq.X(q[0]).with_tags("ignore")),
        cirq.CircuitOperation(cirq.FrozenCircuit(h_cz_y)).repeat(6).with_tags("ignore"),
        _wrap_in_matrix_gate([cirq.CNOT(*q), cirq.CNOT(*q)]),
        cirq.CircuitOperation(cirq.FrozenCircuit(_wrap_in_matrix_gate(h_cz_y))).repeat(4),
        _wrap_in_matrix_gate([cirq.CNOT(*q), cirq.CZ(*q), cirq.CNOT(*q)]),
        cirq.CircuitOperation(cirq.FrozenCircuit(_wrap_in_matrix_gate(h_cz_y)))
        .repeat(5)
        .with_tags("preserve_tag"),
        strategy=cirq.InsertStrategy.NEW,
    )
    c_new_matrix = cirq.merge_k_qubit_unitaries(c_orig, k=2, context=context)
    cirq.testing.assert_same_circuits(c_new_matrix, c_expected_matrix)


def test_merge_k_qubit_unitaries_deep_recurses_on_large_circuit_op():
    q = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(
        cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q[0]), cirq.H(q[0]), cirq.CNOT(*q)))
    )
    c_expected = cirq.Circuit(
        cirq.CircuitOperation(
            cirq.FrozenCircuit(
                cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q[0]), cirq.H(q[0]))).with_tags(
                    "merged"
                ),
                cirq.CNOT(*q),
            )
        )
    )
    c_new = cirq.merge_k_qubit_unitaries(
        c_orig,
        context=cirq.TransformerContext(deep=True),
        k=1,
        rewriter=lambda op: op.with_tags("merged"),
    )
    cirq.testing.assert_same_circuits(c_new, c_expected)
