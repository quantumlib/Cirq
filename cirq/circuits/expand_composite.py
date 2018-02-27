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

"""An optimizer that expands CompositeGates into their constituent gates."""

from typing import Optional

from cirq import ops
from cirq.circuits import Circuit
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.circuits.optimization_pass import PointOptimizer
from cirq.extension.extensions import Extensions


class ExpandComposite(PointOptimizer):
    """An optimization pass that expands CompositeGates.

    For each operation in the circuit, this pass examines if the operation's
    gate is a CompositeGate, or is composite according to a supplied Extension,
    and if it is, clears the operation and replaces it with the composite
    gate elements using a fixed insertion strategy.

    """

    def __init__(self,
        insert_strategy: InsertStrategy = InsertStrategy.INLINE,
        composite_gate_extension: Extensions = Extensions()):
        """Construct the optimization pass.

        Args:
            insert_strategy: The InsertionStrategy used in inserting
                the new gates into the circuit.
            composite_gate_extension: An extension that that can be used
                to supply or override a CompositeGate decomposition.
        """
        self.insert_strategy = insert_strategy
        self.extension = composite_gate_extension

    def optimize_at(self, circuit: Circuit, index: int,
        op: ops.Operation) -> Optional[int]:
        decomposition = self._decompose(op)
        if decomposition is op:
            return

        circuit.clear_operations_touching(op.qubits, (index, ))
        circuit.insert(index, decomposition, strategy=self.insert_strategy)

    def _decompose(self, op):
        """Recursively decompose a composite into an OP_TREE of constituents."""
        composite_gate = self.extension.try_cast(op.gate, ops.CompositeGate)
        if composite_gate is None:
            return op
        return (self._decompose(op) for op in
               composite_gate.default_decompose(op.qubits))
