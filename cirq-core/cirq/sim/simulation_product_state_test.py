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
from typing import Any, Dict, Optional, Sequence

import cirq


class EmptyQuantumState(cirq.QuantumStateRepresentation):
    def copy(self, deep_copy_buffers=True):
        return self

    def measure(self, axes, seed=None):
        return [0] * len(axes)

    @property
    def supports_factor(self):
        return True

    def kron(self, other):
        return self

    def factor(self, axes, *, validate=True, atol=1e-07):
        return self, self

    def reindex(self, axes):
        return self


class EmptySimulationState(cirq.SimulationState):
    def __init__(self, qubits, classical_data):
        super().__init__(state=EmptyQuantumState(), qubits=qubits, classical_data=classical_data)

    def _act_on_fallback_(
        self, action: Any, qubits: Sequence['cirq.Qid'], allow_decompose: bool = True
    ) -> bool:
        return True


q0, q1, q2 = qs3 = cirq.LineQubit.range(3)
qs2 = cirq.LineQubit.range(2)


def create_container(
    qubits: Sequence['cirq.Qid'], split_untangled_states=True
) -> cirq.SimulationProductState[EmptySimulationState]:
    state_map: Dict[Optional['cirq.Qid'], EmptySimulationState] = {}
    log = cirq.ClassicalDataDictionaryStore()
    if split_untangled_states:
        for q in reversed(qubits):
            state_map[q] = EmptySimulationState([q], log)
        state_map[None] = EmptySimulationState((), log)
    else:
        state = EmptySimulationState(qubits, log)
        for q in qubits:
            state_map[q] = state
        state_map[None] = state if not split_untangled_states else EmptySimulationState((), log)
    return cirq.SimulationProductState(
        state_map, qubits, split_untangled_states, classical_data=log
    )


def test_entanglement_causes_join():
    state = create_container(qs2)
    assert len(set(state.values())) == 3
    state.apply_operation(cirq.CNOT(q0, q1))
    assert len(set(state.values())) == 2
    assert state[q0] is state[q1]
    assert state[None] is not state[q0]


def test_subcircuit_entanglement_causes_join():
    state = create_container(qs2)
    assert len(set(state.values())) == 3
    state.apply_operation(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.CNOT(q0, q1))))
    assert len(set(state.values())) == 2
    assert state[q0] is state[q1]


def test_subcircuit_entanglement_causes_join_in_subset():
    state = create_container(qs3)
    assert len(set(state.values())) == 4
    state.apply_operation(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.CNOT(q0, q1))))
    assert len(set(state.values())) == 3
    assert state[q0] is state[q1]
    state.apply_operation(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.CNOT(q0, q2))))
    assert len(set(state.values())) == 2
    assert state[q0] is state[q1] is state[q2]


def test_identity_does_not_join():
    state = create_container(qs2)
    assert len(set(state.values())) == 3
    state.apply_operation(cirq.IdentityGate(2)(q0, q1))
    assert len(set(state.values())) == 3
    assert state[q0] is not state[q1]
    assert state[q0] is not state[None]


def test_identity_fallback_does_not_join():
    state = create_container(qs2)
    assert len(set(state.values())) == 3
    state._act_on_fallback_(cirq.I, (q0, q1))
    assert len(set(state.values())) == 3
    assert state[q0] is not state[q1]
    assert state[q0] is not state[None]


def test_subcircuit_identity_does_not_join():
    state = create_container(qs2)
    assert len(set(state.values())) == 3
    state.apply_operation(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.IdentityGate(2)(q0, q1))))
    assert len(set(state.values())) == 3
    assert state[q0] is not state[q1]


def test_measurement_causes_split():
    state = create_container(qs2)
    state.apply_operation(cirq.CNOT(q0, q1))
    assert len(set(state.values())) == 2
    state.apply_operation(cirq.measure(q0))
    assert len(set(state.values())) == 3
    assert state[q0] is not state[q1]
    assert state[q0] is not state[None]


def test_subcircuit_measurement_causes_split():
    state = create_container(qs2)
    state.apply_operation(cirq.CNOT(q0, q1))
    assert len(set(state.values())) == 2
    state.apply_operation(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(q0))))
    assert len(set(state.values())) == 3
    assert state[q0] is not state[q1]


def test_subcircuit_measurement_causes_split_in_subset():
    state = create_container(qs3)
    state.apply_operation(cirq.CNOT(q0, q1))
    state.apply_operation(cirq.CNOT(q0, q2))
    assert len(set(state.values())) == 2
    state.apply_operation(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(q0))))
    assert len(set(state.values())) == 3
    assert state[q0] is not state[q1]
    state.apply_operation(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(q1))))
    assert len(set(state.values())) == 4
    assert state[q0] is not state[q1]
    assert state[q0] is not state[q2]
    assert state[q1] is not state[q2]


def test_reset_causes_split():
    state = create_container(qs2)
    state.apply_operation(cirq.CNOT(q0, q1))
    assert len(set(state.values())) == 2
    state.apply_operation(cirq.reset(q0))
    assert len(set(state.values())) == 3
    assert state[q0] is not state[q1]
    assert state[q0] is not state[None]


def test_measurement_does_not_split_if_disabled():
    state = create_container(qs2, False)
    state.apply_operation(cirq.CNOT(q0, q1))
    assert len(set(state.values())) == 1
    state.apply_operation(cirq.measure(q0))
    assert len(set(state.values())) == 1
    assert state[q1] is state[q0]
    assert state[None] is state[q0]


def test_reset_does_not_split_if_disabled():
    state = create_container(qs2, False)
    state.apply_operation(cirq.CNOT(q0, q1))
    assert len(set(state.values())) == 1
    state.apply_operation(cirq.reset(q0))
    assert len(set(state.values())) == 1
    assert state[q1] is state[q0]
    assert state[None] is state[q0]


def test_measurement_of_all_qubits_causes_split():
    state = create_container(qs2)
    state.apply_operation(cirq.CNOT(q0, q1))
    assert len(set(state.values())) == 2
    state.apply_operation(cirq.measure(q0, q1))
    assert len(set(state.values())) == 3
    assert state[q0] is not state[q1]
    assert state[q0] is not state[None]


def test_measurement_in_single_qubit_circuit_passes():
    state = create_container([q0])
    assert len(set(state.values())) == 2
    state.apply_operation(cirq.measure(q0))
    assert len(set(state.values())) == 2
    assert state[q0] is not state[None]


def test_reorder_succeeds():
    state = create_container(qs2, False)
    reordered = state[q0].transpose_to_qubit_order([q1, q0])
    assert reordered.qubits == (q1, q0)


def test_copy_succeeds():
    state = create_container(qs2, False)
    copied = state[q0].copy()
    assert copied.qubits == (q0, q1)


def test_merge_succeeds():
    state = create_container(qs2, False)
    merged = state.create_merged_state()
    assert merged.qubits == (q0, q1)


def test_swap_does_not_merge():
    state = create_container(qs2)
    old_q0 = state[q0]
    old_q1 = state[q1]
    state.apply_operation(cirq.SWAP(q0, q1))
    assert len(set(state.values())) == 3
    assert state[q0] is not old_q0
    assert state[q1] is old_q0
    assert state[q1] is not old_q1
    assert state[q0] is old_q1
    assert state[q0].qubits == (q0,)
    assert state[q1].qubits == (q1,)


def test_half_swap_does_merge():
    state = create_container(qs2)
    state.apply_operation(cirq.SWAP(q0, q1) ** 0.5)
    assert len(set(state.values())) == 2
    assert state[q0] is state[q1]


def test_swap_after_entangle_reorders():
    state = create_container(qs2)
    state.apply_operation(cirq.CX(q0, q1))
    assert len(set(state.values())) == 2
    assert state[q0].qubits == (q0, q1)
    state.apply_operation(cirq.SWAP(q0, q1))
    assert len(set(state.values())) == 2
    assert state[q0] is state[q1]
    assert state[q0].qubits == (q1, q0)


def test_act_on_gate_does_not_join():
    state = create_container(qs2)
    assert len(set(state.values())) == 3
    cirq.act_on(cirq.X, state, [q0])
    assert len(set(state.values())) == 3
    assert state[q0] is not state[q1]
    assert state[q0] is not state[None]


def test_field_getters():
    state = create_container(qs2)
    assert state.sim_states.keys() == set(qs2) | {None}
    assert state.split_untangled_states
