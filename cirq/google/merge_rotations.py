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

from cirq import ops, protocols, linalg
from cirq.circuits import (
    Circuit,
    PointOptimizer,
    PointOptimizationSummary,
)
from cirq.google import convert_to_xmon_gates
from cirq.google.decompositions import single_qubit_matrix_to_native_gates
from cirq.google.xmon_gates import XmonGate


class MergeRotations(PointOptimizer):
    """Combines adjacent constant single-qubit rotations."""

    def __init__(self, tolerance: float = 1e-8) -> None:
        super().__init__()
        self.tolerance = tolerance

    def optimization_at(self, circuit, index, op):
        if len(op.qubits) != 1:
            return

        indices, operations = self._scan_single_qubit_ops(
            circuit, index, op.qubits[0])
        if not operations or (len(operations) == 1 and
                              XmonGate.is_supported_op(operations[0])):
            return

        # Replace the gates with a max-2-op XY + Z construction.
        new_operations = self._merge_rotations(op.qubits[0], operations)

        converter = convert_to_xmon_gates.ConvertToXmonGates()
        new_xmon_operations = [converter.convert(new_op)
                               for new_op in new_operations]

        return PointOptimizationSummary(
            clear_span=max(indices) + 1 - index,
            clear_qubits=op.qubits,
            new_operations=new_xmon_operations)

    def _scan_single_qubit_ops(
            self,
            circuit: Circuit,
            index: Optional[int],
            qubit: ops.QubitId) -> Tuple[List[int], List[ops.Operation]]:
        operations = []  # type: List[ops.Operation]
        indices = []  # type: List[int]
        while index is not None:
            op = cast(ops.Operation, circuit.operation_at(qubit, index))
            if len(op.qubits) != 1:
                break
            if protocols.unitary(op, None) is None:
                break
            indices.append(index)
            operations.append(op)
            index = circuit.next_moment_operating_on([qubit], index + 1)
        return indices, operations

    def _merge_rotations(
            self,
            qubit: ops.QubitId,
            operations: Iterable[ops.Operation]
    ) -> List[ops.Operation]:
        matrix = linalg.dot(
            np.eye(2, dtype=np.complex128),
            *reversed([protocols.unitary(op) for op in operations]))

        out_gates = single_qubit_matrix_to_native_gates(matrix, self.tolerance)
        return [gate(qubit) for gate in out_gates]
