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

"""An optimization pass that combines adjacent single-qubit rotations."""

import numpy as np
from typing import List, Tuple, Optional

from cirq import linalg
from cirq import ops
from cirq.circuits.circuit import Circuit
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.circuits.optimization_pass import PointOptimizer
from cirq.circuits import util
from cirq.extension import Extensions


class MergeInteractions(PointOptimizer):
    """Combines adjacent constant single-qubit rotations."""

    def __init__(self,
                 insert_strategy: InsertStrategy = InsertStrategy.INLINE,
                 tolerance: float = 1e-8,
                 allow_partial_czs: bool = True,
                 extensions: Extensions = Extensions()):
        self.insert_strategy = insert_strategy
        self.tolerance = tolerance
        self.allow_partial_czs = allow_partial_czs
        self.extensions = extensions

    def optimize_at(self, circuit, index, op):
        if len(op.qubits) != 2:
            return

        interaction_count, indices, matrix = (
            self._scan_two_qubit_ops_into_matrix(circuit, index, op.qubits))
        if interaction_count <= 1:
            return

        # Find a max-3-cz construction.
        operations = util.two_qubit_matrix_to_native_gates(
            op.qubits[0],
            op.qubits[1],
            matrix,
            self.allow_partial_czs,
            self.tolerance)

        # TODO: don't replace if there's no benefit in CZ depth.
        # Replace the operation.
        circuit.clear_operations_touching(op.qubits, indices)
        return circuit.insert(index + 1, operations, self.insert_strategy)

    def _op_to_matrix(self,
                      op: ops.Operation,
                      qubits: Tuple[ops.QubitId, ...]
                      ) -> Optional[Tuple[np.ndarray, bool]]:
        """Determines the effect of an operation on the given qubits.

        The operation must be a 1-qubit operation on one of the given qubits,
        or a 2-qubit operation on both of the given qubits. Also, the operation
        must have a known matrix. Otherwise None is returned.

        Args:
            op: The operation to understand.
            qubits: The qubits we care about. Order determines matrix tensor
                order.

        Returns:
            None, or else a tuple containing a matrix equivalent to the effect
            of the operation and a boolean indicating if the operation is a
            2-qubit interaction.
        """
        q1, q2 = qubits

        known = self.extensions.try_cast(op.gate, ops.KnownMatrixGate)
        if known is None:
            return None
        m = known.matrix()

        if op.qubits == qubits:
            return m, True
        if op.qubits == (q2, q1):
            return MergeInteractions._flip_kron_order(m), True
        if op.qubits == (q1,):
            return np.kron(np.eye(2), m), False
        if op.qubits == (q2,):
            return np.kron(m, np.eye(2)), False

        return None

    def _scan_two_qubit_ops_into_matrix(
            self,
            circuit: Circuit,
            index: int,
            qubits: Tuple[ops.QubitId, ...]
    ) -> Tuple[int, List[int], np.ndarray]:
        """Accumulates operations affecting the given pair of qubits.

        The scan terminates when it hits the end of the circuit, finds an
        operation without a known matrix, or finds an operation that interacts
        the given qubits with other qubits.

        Args:
            circuit: The circuit to scan for operations.
            index: The index to start scanning forward from.
            qubits: The pair of qubits we care about.

        Returns:
            A tuple containing:
                0. The number of 2-qubit operations that were scanned.
                1. The moment indices those operations were on.
                2. A matrix equivalent to the effect of the scanned operations.
        """

        product = np.eye(4, dtype=np.complex128)
        interaction_count = 0
        touched_indices = []

        while index is not None:
            operations = {circuit.operation_at(q, index) for q in qubits}
            op_data = [
                self._op_to_matrix(op, qubits) for op in
                operations if op
            ]

            # Stop at any non-constant or non-local interaction.
            if any(e is None for e in op_data):
                break

            for op_mat, interacts in op_data:
                product = np.dot(op_mat, product)
                if interacts:
                    interaction_count += 1

            touched_indices.append(index)
            index = circuit.next_moment_operating_on(qubits, index + 1)

        return interaction_count, touched_indices, product

    @staticmethod
    def _flip_kron_order(mat4x4: np.ndarray) -> np.ndarray:
        """Given M = sum(kron(a_i, b_i)), returns M' = sum(kron(b_i, a_i))."""
        result = np.array([[0] * 4] * 4, dtype=np.complex128)
        order = [0, 2, 1, 3]
        for i in range(4):
            for j in range(4):
                result[order[i], order[j]] = mat4x4[i, j]
        return result
