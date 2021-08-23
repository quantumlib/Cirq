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

from typing import FrozenSet, Sequence, Set, TYPE_CHECKING

from cirq import circuits, devices

from cirq.contrib.acquaintance.executor import AcquaintanceOperation, ExecutionStrategy
from cirq.contrib.acquaintance.mutation_utils import expose_acquaintance_gates
from cirq.contrib.acquaintance.permutation import LogicalIndex, LogicalMapping

if TYPE_CHECKING:
    import cirq


class LogicalAnnotator(ExecutionStrategy):
    """Realizes acquaintance opportunities."""

    def __init__(self, initial_mapping: LogicalMapping) -> None:
        """Inits LogicalAnnotator.

        Args:
            initial_mapping: The initial mapping of qubits to logical indices.
        """
        self._initial_mapping = initial_mapping.copy()

    @property
    def initial_mapping(self) -> LogicalMapping:
        return self._initial_mapping

    @property
    def device(self) -> 'cirq.Device':
        return devices.UNCONSTRAINED_DEVICE

    def get_operations(
        self, indices: Sequence[LogicalIndex], qubits: Sequence['cirq.Qid']
    ) -> 'cirq.OP_TREE':
        yield AcquaintanceOperation(qubits, indices)


def get_acquaintance_dag(strategy: 'cirq.Circuit', initial_mapping: LogicalMapping):
    strategy = strategy.copy()
    expose_acquaintance_gates(strategy)
    LogicalAnnotator(initial_mapping)(strategy)
    acquaintance_ops = (
        op
        for moment in strategy._moments
        for op in moment.operations
        if isinstance(op, AcquaintanceOperation)
    )
    return circuits.CircuitDag.from_ops(acquaintance_ops, device=strategy.device)


def get_logical_acquaintance_opportunities(
    strategy: 'cirq.Circuit', initial_mapping: LogicalMapping
) -> Set[FrozenSet[LogicalIndex]]:
    acquaintance_dag = get_acquaintance_dag(strategy, initial_mapping)
    logical_acquaintance_opportunities = set()
    for op in acquaintance_dag.all_operations():
        logical_acquaintance_opportunities.add(frozenset(op.logical_indices))
    return logical_acquaintance_opportunities
