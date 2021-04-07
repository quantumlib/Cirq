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

from typing import DefaultDict, Dict, Sequence, TYPE_CHECKING, Optional

import abc
from collections import defaultdict

from cirq import circuits, devices, ops, protocols

from cirq.contrib.acquaintance.gates import AcquaintanceOpportunityGate
from cirq.contrib.acquaintance.devices import is_acquaintance_strategy
from cirq.contrib.acquaintance.permutation import (
    PermutationGate,
    LogicalIndex,
    LogicalIndexSequence,
    LogicalGates,
    LogicalMapping,
)
from cirq.contrib.acquaintance.mutation_utils import expose_acquaintance_gates

if TYPE_CHECKING:
    import cirq


class ExecutionStrategy(metaclass=abc.ABCMeta):
    """Tells StrategyExecutor how to execute an acquaintance strategy.

    An execution strategy tells StrategyExecutor how to execute an
    acquaintance strategy, i.e. what gates to implement at the available
    acquaintance opportunities."""

    keep_acquaintance = False

    @property
    @abc.abstractmethod
    def device(self) -> 'cirq.Device':
        """The device for which the executed acquaintance strategy should be
        valid.
        """

    @property
    @abc.abstractmethod
    def initial_mapping(self) -> LogicalMapping:
        """The initial mapping of logical indices to qubits."""

    @abc.abstractmethod
    def get_operations(
        self, indices: Sequence[LogicalIndex], qubits: Sequence['cirq.Qid']
    ) -> 'cirq.OP_TREE':
        """Gets the logical operations to apply to qubits."""

    def __call__(self, *args, **kwargs):
        return StrategyExecutor(self)(*args, **kwargs)


class StrategyExecutor(circuits.PointOptimizer):
    """Executes an acquaintance strategy."""

    def __init__(self, execution_strategy: ExecutionStrategy) -> None:
        super().__init__()
        self.execution_strategy = execution_strategy
        self.mapping = execution_strategy.initial_mapping.copy()

    def __call__(self, strategy: 'cirq.Circuit'):
        if not is_acquaintance_strategy(strategy):
            raise TypeError('not is_acquaintance_strategy(strategy)')
        expose_acquaintance_gates(strategy)
        strategy.device = self.execution_strategy.device
        super().optimize_circuit(strategy)
        return self.mapping.copy()

    def optimization_at(
        self, circuit: 'cirq.Circuit', index: int, op: 'cirq.Operation'
    ) -> Optional['cirq.PointOptimizationSummary']:
        if isinstance(op.gate, AcquaintanceOpportunityGate):
            logical_indices = tuple(self.mapping[q] for q in op.qubits)
            logical_operations = self.execution_strategy.get_operations(logical_indices, op.qubits)
            clear_span = int(not self.execution_strategy.keep_acquaintance)

            return circuits.PointOptimizationSummary(
                clear_span=clear_span, clear_qubits=op.qubits, new_operations=logical_operations
            )

        if isinstance(op, ops.GateOperation) and isinstance(op.gate, PermutationGate):
            op.gate.update_mapping(self.mapping, op.qubits)
            return None

        raise TypeError(
            'Can only execute a strategy consisting of gates that '
            'are instances of AcquaintanceOpportunityGate or '
            'PermutationGate.'
        )


class AcquaintanceOperation(ops.GateOperation):
    """Represents an a acquaintance opportunity between a particular set of
    logical indices on a particular set of physical qubits.
    """

    def __init__(
        self, qubits: Sequence['cirq.Qid'], logical_indices: Sequence[LogicalIndex]
    ) -> None:
        if len(logical_indices) != len(qubits):
            raise ValueError('len(logical_indices) != len(qubits)')
        super().__init__(AcquaintanceOpportunityGate(num_qubits=len(qubits)), qubits)
        self.logical_indices: LogicalIndexSequence = logical_indices

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        wire_symbols = tuple(f'({i})' for i in self.logical_indices)
        return protocols.CircuitDiagramInfo(wire_symbols=wire_symbols)


class GreedyExecutionStrategy(ExecutionStrategy):
    """A greedy execution strategy.

    When an acquaintance opportunity is reached, all gates acting on those
    qubits in any order are inserted.
    """

    def __init__(
        self, gates: LogicalGates, initial_mapping: LogicalMapping, device: 'cirq.Device' = None
    ) -> None:
        """
        Args:
            gates: The gates to insert.
            initial_mapping: The initial mapping of qubits to logical indices.
        """

        if len(set(len(indices) for indices in gates)) > 1:
            raise NotImplementedError(
                'Can only implement greedy strategy if all gates are of the same arity.'
            )
        self.index_set_to_gates = self.canonicalize_gates(gates)
        self._initial_mapping = initial_mapping.copy()
        self._device = device or devices.UNCONSTRAINED_DEVICE

    @property
    def initial_mapping(self) -> LogicalMapping:
        return self._initial_mapping

    @property
    def device(self) -> 'cirq.Device':
        return self._device

    def get_operations(
        self, indices: Sequence[LogicalIndex], qubits: Sequence['cirq.Qid']
    ) -> 'cirq.OP_TREE':
        index_set = frozenset(indices)
        if index_set in self.index_set_to_gates:
            gates = self.index_set_to_gates.pop(index_set)
            index_to_qubit = dict(zip(indices, qubits))
            for gate_indices, gate in sorted(gates.items()):
                yield gate(*[index_to_qubit[i] for i in gate_indices])

    @staticmethod
    def canonicalize_gates(gates: LogicalGates) -> Dict[frozenset, LogicalGates]:
        """Canonicalizes a set of gates by the qubits they act on.

        Takes a set of gates specified by ordered sequences of logical
        indices, and groups those that act on the same qubits regardless of
        order."""
        canonicalized_gates: DefaultDict[frozenset, LogicalGates] = defaultdict(dict)
        for indices, gate in gates.items():
            indices = tuple(indices)
            canonicalized_gates[frozenset(indices)][indices] = gate
        return {
            canonical_indices: dict(list(gates.items()))
            for canonical_indices, gates in canonicalized_gates.items()
        }
