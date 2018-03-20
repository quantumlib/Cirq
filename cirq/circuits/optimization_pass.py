# Copyright 2018 Google LLC
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
from typing import Optional, List

import abc
from collections import defaultdict

from cirq import ops
from cirq.circuits.circuit import Circuit


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
    def __init__(self,
                 clear_span: int,
                 clear_qubits: List[ops.QubitId],
                 new_operations: ops.OP_TREE):
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
        return 'PointOptimizationSummary({!r}, {!r}, {!r})'.format(
            self.clear_span,
            self.clear_qubits,
            self.new_operations)


class PointOptimizer:
    """Makes circuit improvements focused on a specific location."""

    @abc.abstractmethod
    def optimization_at(self,
                        circuit: Circuit,
                        index: int,
                        op: ops.Operation
                        ) -> Optional[PointOptimizationSummary]:
        """Rewrites the given circuit to improve it, focusing on one spot.

        Args:
            circuit: The circuit to improve.
            index: The index of the moment with the operation to focus on.
            op: The operation to focus improvements upon.

        Returns:
            The optimization to perform.
        """
        pass

    def optimize_circuit(self, circuit: Circuit):
        walls = defaultdict(lambda: 0)
        i = 0
        while i < len(circuit.moments):  # Note: circuit may mutate as we go.
            for op in circuit.moments[i].operations:
                # Don't touch stuff inserted by previous optimizations.
                if any(walls[q] > i for q in op.qubits):
                    continue

                # Skip if an optimization removed the circuit underneath us.
                if i >= len(circuit.moments):
                    continue
                # Skip if an optimization removed the op we're considering.
                if op not in circuit.moments[i].operations:
                    continue
                opt = self.optimization_at(circuit, i, op)
                # Skip if the optimization did nothing.
                if opt is None:
                    continue

                # Clear target area, and insert new operations.
                circuit.clear_operations_touching(
                    opt.clear_qubits,
                    [e for e in range(i, i + opt.clear_span)])
                next_insert_index = circuit.insert_inline_into_range(
                    opt.new_operations, i, i + opt.clear_span)

                # Prevent redundant optimizations.
                for q in opt.clear_qubits:
                    walls[q] = max(walls[q], next_insert_index)

            i += 1
