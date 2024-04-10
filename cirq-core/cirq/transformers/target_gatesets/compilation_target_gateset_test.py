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

from typing import List
import pytest
import cirq
from cirq.protocols.decompose_protocol import DecomposeResult


def test_compilation_target_gateset():
    class ExampleTargetGateset(cirq.CompilationTargetGateset):
        def __init__(self):
            super().__init__(cirq.AnyUnitaryGateFamily(2))

        @property
        def num_qubits(self) -> int:
            return 2

        def decompose_to_target_gateset(self, op: 'cirq.Operation', _) -> DecomposeResult:
            return op if cirq.num_qubits(op) == 2 and cirq.has_unitary(op) else NotImplemented

        @property
        def preprocess_transformers(self) -> List[cirq.TRANSFORMER]:
            return []

    gateset = ExampleTargetGateset()

    q = cirq.LineQubit.range(2)
    assert cirq.X(q[0]) not in gateset
    assert cirq.CNOT(*q) in gateset
    assert cirq.measure(*q) not in gateset
    circuit_op = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.CZ(*q), cirq.CNOT(*q), cirq.CZ(*q)))
    assert circuit_op in gateset
    assert circuit_op.with_tags(gateset._intermediate_result_tag) not in gateset

    assert gateset.num_qubits == 2
    assert gateset.decompose_to_target_gateset(cirq.X(q[0]), 1) is NotImplemented
    assert gateset.decompose_to_target_gateset(cirq.CNOT(*q), 2) == cirq.CNOT(*q)
    assert gateset.decompose_to_target_gateset(cirq.measure(*q), 3) is NotImplemented

    assert gateset.preprocess_transformers == []
    assert gateset.postprocess_transformers == [
        cirq.merge_single_qubit_moments_to_phxz,
        cirq.drop_negligible_operations,
        cirq.drop_empty_moments,
    ]


class ExampleCXTargetGateset(cirq.TwoQubitCompilationTargetGateset):
    def __init__(self):
        super().__init__(cirq.AnyUnitaryGateFamily(1), cirq.CNOT)

    def _decompose_two_qubit_operation(self, op: 'cirq.Operation', _) -> DecomposeResult:
        if not cirq.has_unitary(op):
            return NotImplemented

        assert self._intermediate_result_tag in op.tags
        q0, q1 = op.qubits
        return [
            cirq.X.on_each(q0, q1),
            cirq.CNOT(q0, q1),
            cirq.Y.on_each(q0, q1),
            cirq.CNOT(q0, q1),
            cirq.Z.on_each(q0, q1),
        ]

    def _decompose_single_qubit_operation(self, op: 'cirq.Operation', _) -> DecomposeResult:
        if not cirq.has_unitary(op):
            return NotImplemented
        assert self._intermediate_result_tag in op.tags
        op_untagged = op.untagged
        assert isinstance(op_untagged, cirq.CircuitOperation)
        return (
            cirq.decompose(op_untagged.circuit)
            if len(op_untagged.circuit) == 1
            else super()._decompose_single_qubit_operation(op, _)
        )


def test_two_qubit_compilation_leaves_single_gates_in_gateset():
    q = cirq.LineQubit.range(2)
    gateset = ExampleCXTargetGateset()

    c = cirq.Circuit(cirq.X(q[0]) ** 0.5)
    cirq.testing.assert_same_circuits(cirq.optimize_for_target_gateset(c, gateset=gateset), c)

    c = cirq.Circuit(cirq.CNOT(*q[:2]))
    cirq.testing.assert_same_circuits(cirq.optimize_for_target_gateset(c, gateset=gateset), c)


def test_two_qubit_compilation_merges_runs_of_single_qubit_gates():
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.CNOT(*q), cirq.X(q[0]), cirq.Y(q[0]), cirq.CNOT(*q))
    cirq.testing.assert_same_circuits(
        cirq.optimize_for_target_gateset(c, gateset=ExampleCXTargetGateset()),
        cirq.Circuit(
            cirq.CNOT(*q),
            cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0, z_exponent=-1).on(q[0]),
            cirq.CNOT(*q),
        ),
    )


def test_two_qubit_compilation_decompose_operation_not_implemented():
    gateset = ExampleCXTargetGateset()
    q = cirq.LineQubit.range(3)
    assert gateset.decompose_to_target_gateset(cirq.measure(q[0]), 1) is NotImplemented
    assert gateset.decompose_to_target_gateset(cirq.measure(*q[:2]), 1) is NotImplemented
    assert (
        gateset.decompose_to_target_gateset(cirq.X(q[0]).with_classical_controls("m"), 1)
        is NotImplemented
    )
    assert gateset.decompose_to_target_gateset(cirq.CCZ(*q), 1) is NotImplemented


def test_two_qubit_compilation_merge_and_replace_to_target_gateset():
    q = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(
        cirq.Moment(cirq.Z(q[1]), cirq.X(q[0])),
        cirq.Moment(cirq.CZ(*q).with_tags("no_compile")),
        cirq.Moment(cirq.Z.on_each(*q)),
        cirq.Moment(cirq.X(q[0])),
        cirq.Moment(cirq.CZ(*q)),
        cirq.Moment(cirq.Z.on_each(*q)),
        cirq.Moment(cirq.X(q[0])),
    )
    cirq.testing.assert_has_diagram(
        c_orig,
        '''
0: ───X───@[no_compile]───Z───X───@───Z───X───
          │                       │
1: ───Z───@───────────────Z───────@───Z───────
''',
    )
    c_new = cirq.optimize_for_target_gateset(
        c_orig,
        gateset=ExampleCXTargetGateset(),
        context=cirq.TransformerContext(tags_to_ignore=("no_compile",)),
    )
    cirq.testing.assert_has_diagram(
        c_new,
        '''
0: ───X───@[no_compile]───X───@───Y───@───Z───
          │                   │       │
1: ───Z───@───────────────X───X───Y───X───Z───
''',
    )


def test_two_qubit_compilation_merge_and_replace_inefficient_component():
    q = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(
        cirq.Moment(cirq.X(q[0])),
        cirq.Moment(cirq.CNOT(*q)),
        cirq.Moment(cirq.X(q[0])),
        cirq.Moment(cirq.CZ(*q).with_tags("no_compile")),
        cirq.Moment(cirq.Z.on_each(*q)),
        cirq.Moment(cirq.X(q[0])),
        cirq.Moment(cirq.CNOT(*q)),
        cirq.Moment(cirq.CNOT(*q)),
        cirq.Moment(cirq.Z.on_each(*q)),
        cirq.Moment(cirq.X(q[0])),
        cirq.Moment(cirq.CNOT(*q)),
        cirq.measure(q[0], key="m"),
        cirq.X(q[1]).with_classical_controls("m"),
    )
    cirq.testing.assert_has_diagram(
        c_orig,
        '''
0: ───X───@───X───@[no_compile]───Z───X───@───@───Z───X───@───M───────
          │       │                       │   │           │   ║
1: ───────X───────@───────────────Z───────X───X───Z───────X───╫───X───
                                                              ║   ║
m: ═══════════════════════════════════════════════════════════@═══^═══
''',
    )
    c_new = cirq.optimize_for_target_gateset(
        c_orig,
        gateset=ExampleCXTargetGateset(),
        context=cirq.TransformerContext(tags_to_ignore=("no_compile",)),
    )
    cirq.testing.assert_has_diagram(
        c_new,
        '''
0: ───X───@───X───@[no_compile]───X───@───Y───@───Z───M───────
          │       │                   │       │       ║
1: ───────X───────@───────────────X───X───Y───X───Z───╫───X───
                                                      ║   ║
m: ═══════════════════════════════════════════════════@═══^═══
''',
    )


def test_two_qubit_compilation_replaces_only_if_2q_gate_count_is_less():
    class ExampleTargetGateset(cirq.TwoQubitCompilationTargetGateset):
        def __init__(self):
            super().__init__(cirq.X, cirq.CNOT)

        def _decompose_two_qubit_operation(self, op: 'cirq.Operation', _) -> DecomposeResult:
            q0, q1 = op.qubits
            return [cirq.X.on_each(q0, q1), cirq.CNOT(q0, q1)] * 10

        def _decompose_single_qubit_operation(self, op: 'cirq.Operation', _) -> DecomposeResult:
            return cirq.X(*op.qubits) if op.gate == cirq.Y else NotImplemented

    q = cirq.LineQubit.range(2)
    ops = [cirq.Y.on_each(*q), cirq.CNOT(*q), cirq.Z.on_each(*q)]
    c_orig = cirq.Circuit(ops)
    c_expected = cirq.Circuit(cirq.X.on_each(*q), ops[-2:])
    c_new = cirq.optimize_for_target_gateset(c_orig, gateset=ExampleTargetGateset())
    cirq.testing.assert_same_circuits(c_new, c_expected)


def test_create_transformer_with_kwargs_raises():
    with pytest.raises(SyntaxError, match="must not contain `context`"):
        cirq.create_transformer_with_kwargs(
            cirq.merge_k_qubit_unitaries, k=2, context=cirq.TransformerContext()
        )
