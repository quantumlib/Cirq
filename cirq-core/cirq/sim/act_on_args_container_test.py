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
from typing import List, Dict, Any, Sequence, Tuple, Optional, Union

import cirq


class EmptyActOnArgs(cirq.ActOnArgs):
    def __init__(self, qubits, logs):
        super().__init__(
            qubits=qubits,
            log_of_measurement_results=logs,
        )

    def _perform_measurement(self, qubits: Sequence[cirq.Qid]) -> List[int]:
        return []

    def copy(self) -> 'EmptyActOnArgs':
        return EmptyActOnArgs(
            qubits=self.qubits,
            logs=self.log_of_measurement_results.copy(),
        )

    def _act_on_fallback_(
        self,
        action: Union['cirq.Operation', 'cirq.Gate'],
        qubits: Sequence['cirq.Qid'],
        allow_decompose: bool = True,
    ) -> bool:
        return True

    def _on_copy(self, args):
        pass

    def _on_kronecker_product(self, other, target):
        pass

    def _on_transpose_to_qubit_order(self, qubits, target):
        pass

    def _on_factor(self, qubits, extracted, remainder, validate=True, atol=1e-07):
        pass

    def sample(self, qubits, repetitions=1, seed=None):
        pass


q0, q1, q2 = qs3 = cirq.LineQubit.range(3)
qs2 = cirq.LineQubit.range(2)


def create_container(
    qubits: Sequence['cirq.Qid'],
    split_untangled_states=True,
) -> cirq.ActOnArgsContainer[EmptyActOnArgs]:
    args_map: Dict[Optional['cirq.Qid'], EmptyActOnArgs] = {}
    log: Dict[str, Any] = {}
    if split_untangled_states:
        for q in reversed(qubits):
            args_map[q] = EmptyActOnArgs([q], log)
        args_map[None] = EmptyActOnArgs((), log)
    else:
        args = EmptyActOnArgs(qubits, log)
        for q in qubits:
            args_map[q] = args
        args_map[None] = args if not split_untangled_states else EmptyActOnArgs((), log)
    return cirq.ActOnArgsContainer(args_map, qubits, split_untangled_states, log)


def test_entanglement_causes_join():
    args = create_container(qs2)
    assert len(set(args.values())) == 3
    args.apply_operation(cirq.CNOT(q0, q1))
    assert len(set(args.values())) == 2
    assert args[q0] is args[q1]
    assert args[None] is not args[q0]


def test_subcircuit_entanglement_causes_join():
    args = create_container(qs2)
    assert len(set(args.values())) == 3
    args.apply_operation(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.CNOT(q0, q1))))
    assert len(set(args.values())) == 2
    assert args[q0] is args[q1]


def test_subcircuit_entanglement_causes_join_in_subset():
    args = create_container(qs3)
    assert len(set(args.values())) == 4
    args.apply_operation(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.CNOT(q0, q1))))
    assert len(set(args.values())) == 3
    assert args[q0] is args[q1]
    args.apply_operation(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.CNOT(q0, q2))))
    assert len(set(args.values())) == 2
    assert args[q0] is args[q1] is args[q2]


def test_identity_does_not_join():
    args = create_container(qs2)
    assert len(set(args.values())) == 3
    args.apply_operation(cirq.IdentityGate(2)(q0, q1))
    assert len(set(args.values())) == 3
    assert args[q0] is not args[q1]
    assert args[q0] is not args[None]


def test_identity_fallback_does_not_join():
    args = create_container(qs2)
    assert len(set(args.values())) == 3
    args._act_on_fallback_(cirq.I, (q0, q1))
    assert len(set(args.values())) == 3
    assert args[q0] is not args[q1]
    assert args[q0] is not args[None]


def test_subcircuit_identity_does_not_join():
    args = create_container(qs2)
    assert len(set(args.values())) == 3
    args.apply_operation(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.IdentityGate(2)(q0, q1))))
    assert len(set(args.values())) == 3
    assert args[q0] is not args[q1]


def test_measurement_causes_split():
    args = create_container(qs2)
    args.apply_operation(cirq.CNOT(q0, q1))
    assert len(set(args.values())) == 2
    args.apply_operation(cirq.measure(q0))
    assert len(set(args.values())) == 3
    assert args[q0] is not args[q1]
    assert args[q0] is not args[None]


def test_subcircuit_measurement_causes_split():
    args = create_container(qs2)
    args.apply_operation(cirq.CNOT(q0, q1))
    assert len(set(args.values())) == 2
    args.apply_operation(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(q0))))
    assert len(set(args.values())) == 3
    assert args[q0] is not args[q1]


def test_subcircuit_measurement_causes_split_in_subset():
    args = create_container(qs3)
    args.apply_operation(cirq.CNOT(q0, q1))
    args.apply_operation(cirq.CNOT(q0, q2))
    assert len(set(args.values())) == 2
    args.apply_operation(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(q0))))
    assert len(set(args.values())) == 3
    assert args[q0] is not args[q1]
    args.apply_operation(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(q1))))
    assert len(set(args.values())) == 4
    assert args[q0] is not args[q1]
    assert args[q0] is not args[q2]
    assert args[q1] is not args[q2]


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


def test_reorder_succeeds():
    args = create_container(qs2, False)
    reordered = args[q0].transpose_to_qubit_order([q1, q0])
    assert reordered.qubits == (q1, q0)


def test_copy_succeeds():
    args = create_container(qs2, False)
    copied = args[q0].copy()
    assert copied.qubits == (q0, q1)


def test_merge_succeeds():
    args = create_container(qs2, False)
    merged = args.create_merged_state()
    assert merged.qubits == (q0, q1)


def test_swap_does_not_merge():
    args = create_container(qs2)
    old_q0 = args[q0]
    old_q1 = args[q1]
    args.apply_operation(cirq.SWAP(q0, q1))
    assert len(set(args.values())) == 3
    assert args[q0] is not old_q0
    assert args[q1] is old_q0
    assert args[q1] is not old_q1
    assert args[q0] is old_q1
    assert args[q0].qubits == (q0,)
    assert args[q1].qubits == (q1,)


def test_half_swap_does_merge():
    args = create_container(qs2)
    args.apply_operation(cirq.SWAP(q0, q1) ** 0.5)
    assert len(set(args.values())) == 2
    assert args[q0] is args[q1]


def test_swap_after_entangle_reorders():
    args = create_container(qs2)
    args.apply_operation(cirq.CX(q0, q1))
    assert len(set(args.values())) == 2
    assert args[q0].qubits == (q0, q1)
    args.apply_operation(cirq.SWAP(q0, q1))
    assert len(set(args.values())) == 2
    assert args[q0] is args[q1]
    assert args[q0].qubits == (q1, q0)


def test_act_on_gate_does_not_join():
    args = create_container(qs2)
    assert len(set(args.values())) == 3
    cirq.act_on(cirq.X, args, [q0])
    assert len(set(args.values())) == 3
    assert args[q0] is not args[q1]
    assert args[q0] is not args[None]
