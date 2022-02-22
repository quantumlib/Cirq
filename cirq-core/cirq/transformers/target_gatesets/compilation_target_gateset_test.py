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
import cirq
from cirq.protocols.decompose_protocol import DecomposeResult


def test_compilation_target_gateset():
    class DummyTargetGateset(cirq.CompilationTargetGateset):
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

    gateset = DummyTargetGateset()

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
