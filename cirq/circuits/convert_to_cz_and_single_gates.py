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

from typing import Optional

from cirq import ops, decompositions, extension, protocols
from cirq.circuits.circuit import Circuit
from cirq.circuits.optimization_pass import (
    PointOptimizationSummary,
    PointOptimizer,
)


class ConvertToCzAndSingleGates(PointOptimizer):
    """Attempts to convert strange multi-qubit gates into CZ and single qubit
    gates.

    First, checks if the given extensions are able to cast the operation into a
        KnownMatrix. If so, and the gate is a 1-qubit or 2-qubit gate, then
        performs circuit synthesis of the operation.

    Second, checks if the given extensions are able to cast the operation into a
        CompositeOperation. If so, recurses on the decomposition.

    Third, if ignore_failures is set, gives up and returns the gate unchanged.
        Otherwise raises a TypeError.
    """

    def __init__(self,
                 extensions: extension.Extensions = None,
                 ignore_failures: bool = False,
                 allow_partial_czs: bool = False) -> None:
        """
        Args:
            extensions: The extensions instance to use when trying to
                cast gates to known types.
            ignore_failures: If set, gates that fail to convert are forwarded
                unchanged. If not set, conversion failures raise a TypeError.
        """
        self.extensions = extensions or extension.Extensions()
        self.ignore_failures = ignore_failures
        self.allow_partial_czs = allow_partial_czs

    def _convert_one(self, op: ops.Operation) -> ops.OP_TREE:
        # Check if this is a CZ
        # Only keep partial CZ gates if allow_partial_czs
        if (isinstance(op, ops.GateOperation)
            and isinstance(op.gate, ops.Rot11Gate)
            and (self.allow_partial_czs or op.gate.half_turns == 1)):
            return op

        # Known matrix?
        mat = protocols.maybe_unitary_effect(op)
        if mat is not None and len(op.qubits) == 1:
            return op
        if mat is not None and len(op.qubits) == 2:
            return decompositions.two_qubit_matrix_to_operations(
                op.qubits[0],
                op.qubits[1],
                mat,
                allow_partial_czs=False)

        # Provides a decomposition?
        composite_op = self.extensions.try_cast(ops.CompositeOperation, op)
        if composite_op is not None:
            return composite_op.default_decompose()

        # Just let it be?
        if self.ignore_failures:
            return op

        raise TypeError("Don't know how to work with {!r}. "
                        "It isn't a 1-qubit KnownMatrix, "
                        "a 2-qubit KnownMatrix, "
                        "or a CompositeOperation.".format(op))

    def convert(self, op: ops.Operation) -> ops.OP_TREE:
        converted = self._convert_one(op)
        if converted is op:
            return converted
        return [self.convert(e) for e in ops.flatten_op_tree(converted)]

    def optimization_at(self, circuit: Circuit, index: int, op: ops.Operation
                        ) -> Optional[PointOptimizationSummary]:
        converted = self.convert(op)
        if converted is op:
            return None

        return PointOptimizationSummary(
            clear_span=1,
            new_operations=converted,
            clear_qubits=op.qubits)
