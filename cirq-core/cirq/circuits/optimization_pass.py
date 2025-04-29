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

from __future__ import annotations

import abc
from collections import defaultdict
from typing import Callable, cast, Dict, Iterable, Optional, Sequence, Tuple, TYPE_CHECKING

from cirq import ops

if TYPE_CHECKING:
    import cirq


class PointOptimizationSummary:
    """A description of a local optimization to perform."""

    def __init__(
        self,
        clear_span: int,
        clear_qubits: Iterable[cirq.Qid],
        new_operations: cirq.OP_TREE,
        preserve_moments: bool = False,
    ) -> None:
        """Inits PointOptimizationSummary.

        Args:
            clear_span: Defines the range of moments to affect. Specifically,
                refers to the indices in range(start, start+clear_span) where
                start is an index known from surrounding context.
            clear_qubits: Defines the set of qubits that should be cleared
                with each affected moment.
            new_operations: The operations to replace the cleared out
                operations with.
            preserve_moments: If set, `cirq.Moment` instances within
                `new_operations` will be preserved exactly. Normally the
                operations would be repacked to fit better into the
                target space, which may move them between moments.
                Please be advised that a PointOptimizer consuming this
                summary will flatten operations no matter what,
                see https://github.com/quantumlib/Cirq/issues/2406.
        """
        self.new_operations = tuple(
            ops.flatten_op_tree(new_operations, preserve_moments=preserve_moments)
        )
        self.clear_span = clear_span
        self.clear_qubits = tuple(clear_qubits)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.clear_span == other.clear_span
            and self.clear_qubits == other.clear_qubits
            and self.new_operations == other.new_operations
        )

    def __ne__(self, other):
        return not self == other

    def __hash__(self) -> int:
        return hash(
            (PointOptimizationSummary, self.clear_span, self.clear_qubits, self.new_operations)
        )

    def __repr__(self) -> str:
        return (
            f'cirq.PointOptimizationSummary({self.clear_span!r}, '
            f'{self.clear_qubits!r}, {self.new_operations!r})'
        )


class PointOptimizer:
    """Makes circuit improvements focused on a specific location."""

    def __init__(
        self,
        post_clean_up: Callable[[Sequence[cirq.Operation]], cirq.OP_TREE] = lambda op_list: op_list,
    ) -> None:
        """Inits PointOptimizer.

        Args:
            post_clean_up: This function is called on each set of optimized
                operations before they are put into the circuit to replace the
                old operations.
        """
        self.post_clean_up = post_clean_up

    def __call__(self, circuit: cirq.Circuit):
        return self.optimize_circuit(circuit)

    @abc.abstractmethod
    def optimization_at(
        self, circuit: cirq.Circuit, index: int, op: cirq.Operation
    ) -> Optional[cirq.PointOptimizationSummary]:
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

    def optimize_circuit(self, circuit: cirq.Circuit):
        frontier: Dict[cirq.Qid, int] = defaultdict(lambda: 0)
        i = 0
        while i < len(circuit):  # Note: circuit may mutate as we go.
            for op in circuit[i].operations:
                # Don't touch stuff inserted by previous optimizations.
                if any(frontier[q] > i for q in op.qubits):
                    continue

                # Skip if an optimization removed the circuit underneath us.
                if i >= len(circuit):
                    continue  # pragma: no cover
                # Skip if an optimization removed the op we're considering.
                if op not in circuit[i].operations:
                    continue  # pragma: no cover
                opt = self.optimization_at(circuit, i, op)
                # Skip if the optimization did nothing.
                if opt is None:
                    continue  # pragma: no cover

                # Clear target area, and insert new operations.
                circuit.clear_operations_touching(
                    opt.clear_qubits, [e for e in range(i, i + opt.clear_span)]
                )
                new_operations = self.post_clean_up(cast(Tuple[ops.Operation], opt.new_operations))

                flat_new_operations = tuple(ops.flatten_to_ops(new_operations))

                new_qubits = set()
                for flat_op in flat_new_operations:
                    for q in flat_op.qubits:
                        new_qubits.add(q)

                if not new_qubits.issubset(set(opt.clear_qubits)):
                    raise ValueError(
                        'New operations in PointOptimizer should not act on new qubits.'
                    )

                circuit.insert_at_frontier(flat_new_operations, i, frontier)
            i += 1
