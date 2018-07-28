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

"""An optimization pass that combines adjacent single-qubit rotations."""

from typing import Iterable, List, Tuple, cast, Optional

import numpy as np

from cirq import ops, extension
from cirq.circuits.circuit import Circuit
from cirq.circuits.optimization_pass import (
    PointOptimizationSummary,
    PointOptimizer,
)


class MergeSingleQubitGates(PointOptimizer):
    """Combines adjacent constant single-qubit rotations into
    SingleQubitMatrixGates.
    """

    def __init__(self,
                 extensions: extension.Extensions = None) -> None:
        self.extensions = extensions or extension.Extensions()

    def optimization_at(self,
                        circuit: Circuit,
                        index: int,
                        op: ops.Operation
                        ) -> Optional[PointOptimizationSummary]:
        if len(op.qubits) != 1:
            return

        indices, gates = self._scan_single_qubit_ops(circuit, index,
                                                     op.qubits[0])
        if not gates:
            return

        # Replace the gates with a max-2-op XY + Z construction.
        operations = self._merge_rotations(op.qubits[0], gates)

        return PointOptimizationSummary(
            clear_span=max(indices) + 1 - index,
            clear_qubits=op.qubits,
            new_operations=operations)

    def _scan_single_qubit_ops(self,
                               circuit: Circuit,
                               index: Optional[int],
                               qubit: ops.QubitId
                               ) -> Tuple[List[int], List[ops.KnownMatrix]]:
        operations = []  # type: List[ops.KnownMatrix]
        indices = []  # type: List[int]
        while index is not None:
            op = cast(ops.Operation, circuit.operation_at(qubit, index))
            if len(op.qubits) != 1:
                break
            operation = self.extensions.try_cast(ops.KnownMatrix, op)
            if operation is None:
                break
            indices.append(index)
            operations.append(operation)
            index = circuit.next_moment_operating_on([qubit], index + 1)
        return indices, operations

    def _merge_rotations(self,
                         qubit: ops.QubitId,
                         operations: Iterable[ops.KnownMatrix]
                         ) -> List[ops.Operation]:
        matrix = np.eye(2, dtype=np.complex128)
        for op in operations:
            matrix = np.dot(op.matrix(), matrix)

        return [ops.SingleQubitMatrixGate(matrix)(qubit)]
