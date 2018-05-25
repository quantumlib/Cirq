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

from cirq import ops
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
                 depth: int = None) -> None:
        """Construct the optimization pass.

        Args:
            composite_gate_extension: An extension that that can be used
                to supply or override a CompositeGate decomposition.
            depth: A limit on the recursion depth with which to expand
                operations. If None, there is no limit.
        """
        self.extension = composite_gate_extension or Extensions()
        if depth is not None and depth < 1:
            raise ValueError('Depth must be at least 1.')
        self.depth = depth

    def optimization_at(self, circuit, index, op):
        decomposition = self._decompose(op, self.depth)
        if decomposition is op:
            return None

        return PointOptimizationSummary(
            clear_span=1,
            clear_qubits=op.qubits,
            new_operations=decomposition)

    def _decompose(self, op, depth):
        """Recursively decompose composite gates into an OP_TREE of gates."""
        if depth == 0:
            return op
        composite_gate = self.extension.try_cast(op.gate, ops.CompositeGate)
        if composite_gate is None:
            return op
        if depth is None:
            return (self._decompose(op, None) for op in
                    composite_gate.default_decompose(op.qubits))
        return (self._decompose(op, depth - 1) for op in
                composite_gate.default_decompose(op.qubits))
