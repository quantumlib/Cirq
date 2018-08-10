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

"""An optimizer that expands CompositeOperation instances."""

from typing import Callable

from cirq import extension, ops
from cirq.circuits.optimization_pass import (
    PointOptimizer,
    PointOptimizationSummary,
)


class ExpandComposite(PointOptimizer):
    """An optimization pass that expands CompositeOperation instances.

    For each operation in the circuit, this pass examines if the operation is a
    CompositeOperation, or is composite according to a supplied Extension,
    and if it is, clears the operation and replaces it with its decomposition
    using a fixed insertion strategy.
    """

    def __init__(self,
                 composite_gate_extension: extension.Extensions = None,
                 no_decomp: Callable[[ops.Operation], bool]=(lambda _: False)
                 ) -> None:
        """Construct the optimization pass.

        Args:
            composite_gate_extension: An extension that that can be used
                to supply or override a CompositeOperation decomposition.
            no_decomp: A predicate that determines whether an operation should
                be decomposed or not. Defaults to decomposing everything.
        """
        super().__init__()
        self.extension = composite_gate_extension or extension.Extensions()
        self.no_decomp = no_decomp

    def optimization_at(self, circuit, index, op):
        decomposition = self._decompose(op)
        if decomposition is op:
            return None

        return PointOptimizationSummary(
            clear_span=1,
            clear_qubits=op.qubits,
            new_operations=decomposition)

    def _decompose(self, op: ops.Operation) -> ops.OP_TREE:
        """Recursively decompose composite gates into an OP_TREE of gates."""
        skip = self.no_decomp(op)
        if skip and (skip is not NotImplemented):
            return op
        composite_op = self.extension.try_cast(ops.CompositeOperation, op)
        if composite_op is None:
            return op
        op_iter = ops.flatten_op_tree(composite_op.default_decompose())
        return (self._decompose(op) for op in op_iter)
