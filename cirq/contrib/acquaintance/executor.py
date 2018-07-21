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
from typing import Dict, Tuple, TYPE_CHECKING

from cirq.circuits import (
        Circuit, PointOptimizer, PointOptimizationSummary)
from cirq.ops import Operation, GateOperation, Gate, QubitId
from cirq.contrib.acquaintance.gates import AcquaintanceOpportunityGate
from cirq.contrib.acquaintance.permutation import (
        PermutationGate, LOGICAL_INDEX)
from cirq.contrib.acquaintance.strategy import AcquaintanceStrategy

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import Callable, List, DefaultDict

class StrategyExecutor(PointOptimizer):
    """Executes an AcquaintanceStrategy.

    Args:
        operations: The operations to implement.

    Raises:
        TypeError: Applied to a circuit containing a gate that is not an
            instance of AcquaintanceOpportunityGate or PermutationGate.
    """


    def execute(self,
                strategy: AcquaintanceStrategy,
                gates: Dict[Tuple[LOGICAL_INDEX,...], Gate],
                initial_mapping: Dict[QubitId, LOGICAL_INDEX]
                ) -> None:
        """Executes the strategy.

        Args:
            strategy: The acquaintance strategy.
            gates: The operations to implement.
            initial_mapping: The initial mapping from qubits to logical indices.

        Raises:
            ValueError:
                * The initial mapping specifies qubits not in the strategy.
                * The initial mapping doesn't specify a qubit for some logical
                  index.
            NotImplementedError:
                * The operations are of different arities
                * The arity of the operations doesn't exactly match the arity
                  of the acquaintance opportunities.
        """
        if not set(initial_mapping).issubset(strategy.all_qubits()):
            raise ValueError('Initial mapping specifies qubits not in the'
                    'strategy.')
        all_indices = set(i for indices in gates for i in indices)
        if not all_indices.issubset(initial_mapping.values()):
            raise ValueError('Initial mapping does not specify qubit for '
                             'every logical index.')
        if (set(len(indices) for indices in gates) !=
                set((strategy.acquaintance_size(),))):
            raise NotImplementedError('The arity of the operations must match '
                    'that of the acquaintance opportunities exactly.')
        self.index_set_to_gates = self.canonicalize_gates(gates)
        self.mapping = {q: i for q, i in initial_mapping.items()}
        super().optimize_circuit(strategy)
        if self.index_set_to_gates:
            raise ValueError("Strategy couldn't implement all operations.")

    def optimization_at(self, circuit: Circuit, index: int, op: Operation):
        if (isinstance(op, GateOperation) and
                isinstance(op.gate, AcquaintanceOpportunityGate)):
            index_set = frozenset(self.mapping[q] for q in op.qubits)
            logical_operations = []
            if index_set in self.index_set_to_gates:
                gates = self.index_set_to_gates.pop(index_set)
                index_to_qubit = {self.mapping[q]: q for q in op.qubits}
                for indices, gate in gates.items():
                    op = gate(*[index_to_qubit[i] for i in indices])
                    logical_operations.append(op)
            return PointOptimizationSummary(
                    clear_span=1,
                    clear_qubits=op.qubits,
                    new_operations=logical_operations)

        if (isinstance(op, GateOperation) and
                isinstance(op.gate, PermutationGate)):
            op.gate.update_mapping(self.mapping, op.qubits)

        raise TypeError('Can only execute a strategy consisting of gates that '
                         'are instances of AcquaintanceOpportunityGate or '
                         'PermutationGate.')


    @staticmethod
    def canonicalize_gates(gates: Dict[Tuple[LOGICAL_INDEX, ...], Gate]
        ) -> Dict[frozenset, Dict[Tuple[LOGICAL_INDEX, ...], Gate]]:
        canonicalized_gates = defaultdict(dict
            ) # type: Dict[frozenset, Dict[Tuple[LOGICAL_INDEX, ...], Gate]]
        for indices, gate in gates.items():
            indices = tuple(indices)
            canonicalized_gates[frozenset(indices)][indices] = gate
        return {canonical_indices: dict(list(gates.items()))
                for canonical_indices, gates in canonicalized_gates.items()}
