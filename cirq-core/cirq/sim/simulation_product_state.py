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

from collections import abc
from typing import Any, Dict, Generic, Iterator, List, Mapping, Optional, Sequence, TYPE_CHECKING

import numpy as np

from cirq import ops, protocols, value
from cirq.sim.simulation_state import TSimulationState
from cirq.sim.simulation_state_base import SimulationStateBase

if TYPE_CHECKING:
    import cirq


class SimulationProductState(
    Generic[TSimulationState], SimulationStateBase[TSimulationState], abc.Mapping
):
    """A container for a `Qid`-to-`SimulationState` dictionary."""

    def __init__(
        self,
        sim_states: Dict[Optional['cirq.Qid'], TSimulationState],
        qubits: Sequence['cirq.Qid'],
        split_untangled_states: bool,
        classical_data: Optional['cirq.ClassicalDataStore'] = None,
    ):
        """Initializes the class.

        Args:
            sim_states: The `SimulationState` dictionary. This will not be
                copied; the original reference will be kept here.
            qubits: The canonical ordering of qubits.
            split_untangled_states: If True, optimizes operations by running
                unentangled qubit sets independently and merging those states
                at the end.
            classical_data: The shared classical data container for this
                simulation.
        """
        classical_data = classical_data or value.ClassicalDataDictionaryStore()
        super().__init__(qubits=qubits, classical_data=classical_data)
        self._sim_states = sim_states
        self._split_untangled_states = split_untangled_states

    @property
    def sim_states(self) -> Mapping[Optional['cirq.Qid'], TSimulationState]:
        return self._sim_states

    @property
    def split_untangled_states(self) -> bool:
        return self._split_untangled_states

    def create_merged_state(self) -> TSimulationState:
        if not self.split_untangled_states:
            return self.sim_states[None]
        final_args = self.sim_states[None]
        for args in set([self.sim_states[k] for k in self.sim_states.keys() if k is not None]):
            final_args = final_args.kronecker_product(args)
        return final_args.transpose_to_qubit_order(self.qubits)

    def _act_on_fallback_(
        self, action: Any, qubits: Sequence['cirq.Qid'], allow_decompose: bool = True
    ) -> bool:
        gate_opt = (
            action
            if isinstance(action, ops.Gate)
            else action.gate
            if isinstance(action, ops.Operation)
            else None
        )

        if isinstance(gate_opt, ops.IdentityGate):
            return True

        if (
            isinstance(gate_opt, ops.SwapPowGate)
            and gate_opt.exponent % 2 == 1
            and gate_opt.global_shift == 0
        ):
            q0, q1 = qubits
            args0 = self.sim_states[q0]
            args1 = self.sim_states[q1]
            if args0 is args1:
                args0.swap(q0, q1, inplace=True)
            else:
                self._sim_states[q0] = args1.rename(q1, q0, inplace=True)
                self._sim_states[q1] = args0.rename(q0, q1, inplace=True)
            return True

        # Go through the op's qubits and join any disparate SimulationState states
        # into a new combined state.
        op_args_opt: Optional[TSimulationState] = None
        for q in qubits:
            if op_args_opt is None:
                op_args_opt = self.sim_states[q]
            elif q not in op_args_opt.qubits:
                op_args_opt = op_args_opt.kronecker_product(self.sim_states[q])
        op_args = op_args_opt or self.sim_states[None]

        # (Backfill the args map with the new value)
        for q in op_args.qubits:
            self._sim_states[q] = op_args

        # Act on the args with the operation
        act_on_qubits = qubits if isinstance(action, ops.Gate) else None
        protocols.act_on(action, op_args, act_on_qubits, allow_decompose=allow_decompose)

        # Decouple any measurements or resets
        if self.split_untangled_states and isinstance(
            gate_opt, (ops.ResetChannel, ops.MeasurementGate)
        ):
            for q in qubits:
                if op_args.allows_factoring and len(op_args.qubits) > 1:
                    q_args, op_args = op_args.factor((q,), validate=False)
                    self._sim_states[q] = q_args

            # (Backfill the args map with the new value)
            for q in op_args.qubits:
                self._sim_states[q] = op_args
        return True

    def copy(
        self, deep_copy_buffers: bool = True
    ) -> 'cirq.SimulationProductState[TSimulationState]':
        classical_data = self._classical_data.copy()
        copies = {}
        for sim_state in set(self.sim_states.values()):
            copies[sim_state] = sim_state.copy(deep_copy_buffers)
        for copy in copies.values():
            copy._classical_data = classical_data
        args = {q: copies[a] for q, a in self.sim_states.items()}
        return SimulationProductState(
            args, self.qubits, self.split_untangled_states, classical_data=classical_data
        )

    def sample(
        self,
        qubits: List['cirq.Qid'],
        repetitions: int = 1,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ) -> np.ndarray:
        columns = []
        selected_order: List[ops.Qid] = []
        q_set = set(qubits)
        for v in dict.fromkeys(self.sim_states.values()):
            qs = [q for q in v.qubits if q in q_set]
            if any(qs):
                column = v.sample(qs, repetitions, seed)
                columns.append(column)
                selected_order += qs
        stacked = np.column_stack(columns)
        qubit_map = {q: i for i, q in enumerate(selected_order)}
        index_order = [qubit_map[q] for q in qubits]
        return stacked[:, index_order]

    def __getitem__(self, item: Optional['cirq.Qid']) -> TSimulationState:
        return self.sim_states[item]

    def __len__(self) -> int:
        return len(self.sim_states)

    def __iter__(self) -> Iterator[Optional['cirq.Qid']]:
        return iter(self.sim_states)
