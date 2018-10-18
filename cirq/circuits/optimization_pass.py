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

"""Defines the OptimizationPass type."""
from typing import Callable, Iterable, Optional, Sequence, TYPE_CHECKING

import abc
from collections import defaultdict

from cirq import ops
from cirq.circuits.circuit import Circuit

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from cirq.ops import QubitId
    from typing import Dict


class OptimizationPass:
    """Rewrites a circuit's operations in place to make them better."""

    @abc.abstractmethod
    def optimize_circuit(self, circuit: Circuit):
        """Rewrites the given circuit to make it better.

        Note that this performs an in place optimization.

        Args:
            circuit: The circuit to improve.
        """
        pass


class PointOptimizationSummary:
    """A description of a local optimization to perform."""

    def __init__(self,
                 clear_span: int,
                 clear_qubits: Iterable[ops.QubitId],
                 new_operations: ops.OP_TREE) -> None:
        """
        Args:
            clear_span: Defines the range of moments to affect. Specifically,
                refers to the indices in range(start, start+clear_span) where
                start is an index known from surrounding context.
            clear_qubits: Defines the set of qubits that should be cleared
                with each affected moment.
            new_operations: The operations to replace the cleared out
                operations with.
        """
        self.new_operations = tuple(ops.flatten_op_tree(new_operations))
        self.clear_span = clear_span
        self.clear_qubits = tuple(clear_qubits)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.clear_span == other.clear_span and
                self.clear_qubits == other.clear_qubits and
                self.new_operations == other.new_operations)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((PointOptimizationSummary,
                     self.clear_span,
                     self.clear_qubits,
                     self.new_operations))

    def __repr__(self):
        return 'cirq.PointOptimizationSummary({!r}, {!r}, {!r})'.format(
            self.clear_span,
            self.clear_qubits,
            self.new_operations)


class PointOptimizer(OptimizationPass):
    """Makes circuit improvements focused on a specific location."""

    def __init__(self,
                 post_clean_up: Callable[[Sequence[ops.Operation]], ops.OP_TREE
                                ] = lambda op_list: op_list
                 ) -> None:
        """
        Args:
            post_clean_up: This function is called on each set of optimized
                operations before they are put into the circuit to replace the
                old operations.
        """
        self.post_clean_up = post_clean_up

    @abc.abstractmethod
    def optimization_at(self,
                        circuit: Circuit,
                        index: int,
                        op: ops.Operation
                        ) -> Optional[PointOptimizationSummary]:
        """Describes how to change operations near the given location.

        For example, this method could realize that the given operation is an
        X gate and that in the very next moment there is a Z gate. It would
        indicate that they should be combined into a Y gate by returning
        PointOptimizationSummary(clear_span=2,
                                 clear_qubits=op.qubits,
                                 new_operations=cirq.Y(op.qubits[0]))

        Args:
            circuit: The circuit to improve.
            index: The index of the moment with the operation to focus on.
            op: The operation to focus improvements upon.

        Returns:
            A description of the optimization to perform, or else None if no
            change should be made.
        """
        pass

    def optimize_circuit(self, circuit: Circuit):
        frontier = defaultdict(lambda: 0)  # type: Dict[QubitId, int]
        i = 0
        while i < len(circuit):  # Note: circuit may mutate as we go.
            for op in circuit[i].operations:
                # Don't touch stuff inserted by previous optimizations.
                if any(frontier[q] > i for q in op.qubits):
                    continue

                # Skip if an optimization removed the circuit underneath us.
                if i >= len(circuit):
                    continue
                # Skip if an optimization removed the op we're considering.
                if op not in circuit[i].operations:
                    continue
                opt = self.optimization_at(circuit, i, op)
                # Skip if the optimization did nothing.
                if opt is None:
                    continue

                # Clear target area, and insert new operations.
                circuit.clear_operations_touching(
                    opt.clear_qubits,
                    [e for e in range(i, i + opt.clear_span)])
                new_operations = self.post_clean_up(opt.new_operations)
                circuit.insert_at_frontier(new_operations, i, frontier)

            i += 1

    def __call__(self, circuit: Circuit):
        return self.optimize_circuit(circuit)
