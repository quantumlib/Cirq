# Copyright 2018 The Cirq Developers
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

from collections import defaultdict
from typing import Dict, TYPE_CHECKING, Sequence

from cirq import abc
from cirq.circuits import (
        Circuit, PointOptimizer, PointOptimizationSummary)
from cirq.ops import Operation, GateOperation, QubitId, OP_TREE
from cirq.contrib.acquaintance.gates import AcquaintanceOpportunityGate
from cirq.contrib.acquaintance.permutation import (
        PermutationGate,
        LogicalIndex,
        LogicalGates, LogicalMapping)
from cirq.contrib.acquaintance.strategy import (
        AcquaintanceStrategy, expose_acquaintance_gates)

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import Callable, List, DefaultDict

class ExecutionStrategy(metaclass=abc.ABCMeta):
    """An execution strategy."""

    keep_acquaintance = False

    @abc.abstractproperty
    def initial_mapping(self) -> LogicalMapping:
        pass

    @abc.abstractmethod
    def get_operations(self,
                       indices: Sequence[LogicalIndex],
                       qubits: Sequence[QubitId]
                       ) -> OP_TREE:
        """Gets the logical operations to apply to qubits."""
        pass


class StrategyExecutor(PointOptimizer):
    """Executes an AcquaintanceStrategy."""

    def __init__(self, execution_strategy: ExecutionStrategy) -> None:
        self.execution_strategy = execution_strategy
        self.mapping = execution_strategy.initial_mapping.copy()

    def __call__(self, strategy: Circuit):
        if not isinstance(strategy, AcquaintanceStrategy):
            raise ValueError('Can only execute an acquaintance strategy.')
        expose_acquaintance_gates(strategy)
        super().optimize_circuit(strategy)
        return self.mapping.copy()

    def optimization_at(self, circuit: Circuit, index: int, op: Operation):
        if (isinstance(op, GateOperation) and
                isinstance(op.gate, AcquaintanceOpportunityGate)):
            logical_indices = tuple(self.mapping[q] for q in op.qubits)
            logical_operations = self.execution_strategy.get_operations(
                    logical_indices, op.qubits)
            clear_span = int(not self.execution_strategy.keep_acquaintance)

            return PointOptimizationSummary(
                    clear_span=clear_span,
                    clear_qubits=op.qubits,
                    new_operations=logical_operations)

        if (isinstance(op, GateOperation) and
                isinstance(op.gate, PermutationGate)):
            op.gate.update_mapping(self.mapping, op.qubits)
            return

        raise TypeError('Can only execute a strategy consisting of gates that '
                         'are instances of AcquaintanceOpportunityGate or '
                         'PermutationGate.')

class GreedyExecutionStrategy(ExecutionStrategy):
    def __init__(self,
                 gates: LogicalGates,
                 initial_mapping: LogicalMapping
                 ) -> None:
        if len(set(len(indices) for indices in gates)) > 1:
            raise NotImplementedError(
                    'Can only implement greedy strategy if all gates '
                    'are of the same arity.')
        self.index_set_to_gates = self.canonicalize_gates(gates)
        self._initial_mapping = initial_mapping.copy()

    @property
    def initial_mapping(self) -> LogicalMapping:
        return self._initial_mapping

    def get_operations(self,
                       indices: Sequence[LogicalIndex],
                       qubits: Sequence[QubitId]
                       ) -> OP_TREE:
        index_set = frozenset(indices)
        if index_set in self.index_set_to_gates:
            gates = self.index_set_to_gates.pop(index_set)
            index_to_qubit = dict(zip(indices, qubits))
            for gate_indices, gate in gates.items():
                yield gate(*[index_to_qubit[i] for i in gate_indices])
        return


    @staticmethod
    def canonicalize_gates(gates: LogicalGates
        ) -> Dict[frozenset, LogicalGates]:
        canonicalized_gates = defaultdict(dict
            ) # type: DefaultDict[frozenset, LogicalGates]
        for indices, gate in gates.items():
            indices = tuple(indices)
            canonicalized_gates[frozenset(indices)][indices] = gate
        return {canonical_indices: dict(list(gates.items()))
                for canonical_indices, gates in canonicalized_gates.items()}
