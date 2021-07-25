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
from typing import (
    Dict,
    TYPE_CHECKING,
    Generic,
    Sequence,
    Optional,
    Iterator,
    Any,
    Tuple,
    Set,
    List,
)

import numpy as np

from cirq import ops
from cirq.sim.operation_target import OperationTarget
from cirq.sim.simulator import (
    TActOnArgs,
)

if TYPE_CHECKING:
    import cirq


class ActOnArgsContainer(
    Generic[TActOnArgs],
    OperationTarget[TActOnArgs],
    abc.Mapping,
):
    """A container for a `Qid`-to-`ActOnArgs` dictionary."""

    def __init__(
        self,
        args: Dict[Optional['cirq.Qid'], TActOnArgs],
        qubits: Sequence['cirq.Qid'],
        split_untangled_states: bool,
        log_of_measurement_results: Dict[str, Any],
    ):
        """Initializes the class.

        Args:
            args: The `ActOnArgs` dictionary. This will not be copied; the
                original reference will be kept here.
            qubits: The canonical ordering of qubits.
            split_untangled_states: If True, optimizes operations by running
                unentangled qubit sets independently and merging those states
                at the end.
            log_of_measurement_results: A mutable object that measurements are
                being recorded into.
        """
        self.args = args
        self._qubits = tuple(qubits)
        self.split_untangled_states = split_untangled_states
        self._log_of_measurement_results = log_of_measurement_results

    def create_merged_state(self) -> TActOnArgs:
        if not self.split_untangled_states:
            return self.args[None]
        final_args = self.args[None]
        for args in set([self.args[k] for k in self.args.keys() if k is not None]):
            final_args = final_args.kronecker_product(args)
        return final_args.transpose_to_qubit_order(self.qubits)

    def apply_operation(
        self,
        op: 'cirq.Operation',
    ):
        # Go through the op's qubits and join any disparate ActOnArgs states
        # into a new combined state.
        op_args_opt: Optional[TActOnArgs] = None
        for q in op.qubits:
            if op_args_opt is None:
                op_args_opt = self.args[q]
            elif q not in op_args_opt.qubits:
                op_args_opt = op_args_opt.kronecker_product(self.args[q])
        op_args = op_args_opt or self.args[None]

        # (Backfill the args map with the new value)
        for q in op_args.qubits:
            self.args[q] = op_args

        # Act on the args with the operation
        op_args.apply_operation(op)

        # Decouple any measurements or resets
        if self.split_untangled_states and isinstance(
            op.gate, (ops.MeasurementGate, ops.ResetChannel)
        ):
            for q in op.qubits:
                q_args, op_args = op_args.factor((q,), validate=False)
                self.args[q] = q_args

            # (Backfill the args map with the new value)
            for q in op_args.qubits:
                self.args[q] = op_args

    def copy(self) -> 'ActOnArgsContainer[TActOnArgs]':
        logs = self.log_of_measurement_results.copy()
        copies = {a: a.copy() for a in set(self.args.values())}
        for copy in copies.values():
            copy._log_of_measurement_results = logs
        args = {q: copies[a] for q, a in self.args.items()}
        return ActOnArgsContainer(args, self.qubits, self.split_untangled_states, logs)

    @property
    def qubits(self) -> Tuple['cirq.Qid', ...]:
        return self._qubits

    @property
    def log_of_measurement_results(self) -> Dict[str, Any]:
        return self._log_of_measurement_results

    def sample(
        self,
        qubits: List[ops.Qid],
        repetitions: int = 1,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ) -> np.ndarray:
        columns = []
        selected_order: List[ops.Qid] = []
        q_set = set(qubits)
        for v in dict.fromkeys(self.args.values()):
            qs = [q for q in v.qubits if q in q_set]
            if any(qs):
                column = v.sample(qs, repetitions, seed)
                columns.append(column)
                selected_order += qs
        stacked = np.column_stack(columns)
        qubit_map = {q: i for i, q in enumerate(selected_order)}
        index_order = [qubit_map[q] for q in qubits]
        return stacked[:, index_order]

    def __getitem__(self, item: Optional['cirq.Qid']) -> TActOnArgs:
        return self.args[item]

    def __len__(self) -> int:
        return len(self.args)

    def __iter__(self) -> Iterator[Optional['cirq.Qid']]:
        return iter(self.args)
