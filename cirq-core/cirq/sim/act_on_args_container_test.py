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
from typing import List, Dict, Any, Sequence, Tuple, Optional

import cirq


class TestActOnArgs(cirq.ActOnArgs):
    def __init__(self, qubits, logs):
        super().__init__(
            qubits=qubits,
            log_of_measurement_results=logs,
        )

    def _perform_measurement(self) -> List[int]:
        return []

    def copy(self) -> 'TestActOnArgs':
        return TestActOnArgs(
            qubits=self.qubits,
            logs=self.log_of_measurement_results.copy(),
        )

    def _act_on_fallback_(self, action: Any, allow_decompose: bool):
        return True

    def join(self, other: 'TestActOnArgs') -> 'TestActOnArgs':
        return TestActOnArgs(
            qubits=self.qubits + other.qubits,
            logs=self.log_of_measurement_results,
        )

    def extract(self, qubits: Sequence['cirq.Qid']) -> Tuple['TestActOnArgs', 'TestActOnArgs']:
        extracted_args = TestActOnArgs(
            qubits=qubits,
            logs=self.log_of_measurement_results,
        )
        remainder_args = TestActOnArgs(
            qubits=tuple(q for q in self.qubits if q not in qubits),
            logs=self.log_of_measurement_results,
        )
        return extracted_args, remainder_args

    def reorder(self, qubits: Sequence['cirq.Qid']) -> 'TestActOnArgs':
        return TestActOnArgs(
            qubits=qubits,
            logs=self.log_of_measurement_results,
        )


q0, q1 = qs2 = cirq.LineQubit.range(2)


def create_container(
    qubits: Sequence['cirq.Qid'],
    split_untangled_states=True,
) -> cirq.ActOnArgsContainer[TestActOnArgs]:
    args_map: Dict[Optional['cirq.Qid'], TestActOnArgs] = {}
    log: Dict[str, Any] = {}
    if split_untangled_states:
        for q in reversed(qubits):
            args_map[q] = TestActOnArgs([q], log)
        args_map[None] = TestActOnArgs((), log)
    else:
        args = TestActOnArgs(qubits, log)
        for q in qubits:
            args_map[q] = args
        args_map[None] = args if not split_untangled_states else TestActOnArgs((), log)
    return cirq.ActOnArgsContainer(args_map, qubits, split_untangled_states)


def test_entanglement_causes_join():
    args = create_container(qs2)
    assert len(set(args.values())) == 3
    args.apply_operation(cirq.CNOT(q0, q1))
    assert len(set(args.values())) == 2
    assert args[q0] is args[q1]
    assert args[None] is not args[q0]


def test_measurement_causes_split():
    args = create_container(qs2)
    args.apply_operation(cirq.CNOT(q0, q1))
    assert len(set(args.values())) == 2
    args.apply_operation(cirq.measure(q0))
    assert len(set(args.values())) == 3
    assert args[q0] is not args[q1]
    assert args[q0] is not args[None]


def test_reset_causes_split():
    args = create_container(qs2)
    args.apply_operation(cirq.CNOT(q0, q1))
    assert len(set(args.values())) == 2
    args.apply_operation(cirq.reset(q0))
    assert len(set(args.values())) == 3
    assert args[q0] is not args[q1]
    assert args[q0] is not args[None]


def test_measurement_does_not_split_if_disabled():
    args = create_container(qs2, False)
    args.apply_operation(cirq.CNOT(q0, q1))
    assert len(set(args.values())) == 1
    args.apply_operation(cirq.measure(q0))
    assert len(set(args.values())) == 1
    assert args[q1] is args[q0]
    assert args[None] is args[q0]


def test_reset_does_not_split_if_disabled():
    args = create_container(qs2, False)
    args.apply_operation(cirq.CNOT(q0, q1))
    assert len(set(args.values())) == 1
    args.apply_operation(cirq.reset(q0))
    assert len(set(args.values())) == 1
    assert args[q1] is args[q0]
    assert args[None] is args[q0]


def test_measurement_of_all_qubits_causes_split():
    args = create_container(qs2)
    args.apply_operation(cirq.CNOT(q0, q1))
    assert len(set(args.values())) == 2
    args.apply_operation(cirq.measure(q0, q1))
    assert len(set(args.values())) == 3
    assert args[q0] is not args[q1]
    assert args[q0] is not args[None]


def test_measurement_in_single_qubit_circuit_passes():
    args = create_container([q0])
    assert len(set(args.values())) == 2
    args.apply_operation(cirq.measure(q0))
    assert len(set(args.values())) == 2
    assert args[q0] is not args[None]
