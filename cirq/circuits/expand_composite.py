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

"""An optimizer that expands CompositeGates into their constituent gates."""

from typing import Callable

from cirq.ops import Operation, CompositeGate, flatten_op_tree, OP_TREE
from cirq.circuits.optimization_pass import (
    PointOptimizer,
    PointOptimizationSummary,
)
from cirq.extension import Extensions


class ExpandComposite(PointOptimizer):
    """An optimization pass that expands CompositeGates.

    For each operation in the circuit, this pass examines if the operation's
    gate is a CompositeGate, or is composite according to a supplied Extension,
    and if it is, clears the operation and replaces it with the composite
    gate elements using a fixed insertion strategy.

    """

    def __init__(self,
                 composite_gate_extension: Extensions = None,
                 no_decomp: Callable[[Operation], bool]=(lambda _: False)
                 ) -> None:
        """Construct the optimization pass.

        Args:
            composite_gate_extension: An extension that that can be used
                to supply or override a CompositeGate decomposition.
            no_decomp: A predicate (of an operation) that indicates that it
                should not be decomposed.
        """
        self.extension = composite_gate_extension or Extensions()
        self.no_decomp = no_decomp

    def optimization_at(self, circuit, index, op):
        decomposition = self._decompose(op)
        if decomposition is op:
            return None

        return PointOptimizationSummary(
            clear_span=1,
            clear_qubits=op.qubits,
            new_operations=decomposition)

    def _decompose(self, op: Operation) -> OP_TREE:
        """Recursively decompose composite gates into an OP_TREE of gates."""
        skip = self.no_decomp(op)
        if skip and (skip is not NotImplemented):
            return op
        composite_gate = self.extension.try_cast(CompositeGate, op.gate)
        if composite_gate is None:
            return op
        op_iter = flatten_op_tree(
                        composite_gate.default_decompose(op.qubits))
        return (self._decompose(op) for op in op_iter)
