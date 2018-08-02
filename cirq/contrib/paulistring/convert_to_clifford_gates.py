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

import numpy as np

from cirq import ops, linalg, decompositions, extension, protocols
from cirq.circuits.circuit import Circuit
from cirq.circuits.optimization_pass import (
    PointOptimizationSummary,
    PointOptimizer,
)


class ConvertToCliffordGates(PointOptimizer):
    """Attempts to convert single-qubit gates into single-qubit
    CliffordGates.

    First, checks if the given extensions are able to cast the operation into a
        KnownMatrix. If so, and the gate is a 1-qubit gate, then decomposes it
        and tries to make a CliffordGate. It fails if the operation is not in
        the Clifford group.

    Second, checks if the given extensions are able to cast the operation into a
        CompositeOperation. If so, recurses on the decomposition.
    """

    def __init__(self,
                 ignore_failures: bool = False,
                 tolerance: float = 0,
                 extensions: extension.Extensions = None) -> None:
        """
        Args:
            ignore_failures: If set, gates that fail to convert are forwarded
                unchanged. If not set, conversion failures raise a TypeError.
            tolerance: Maximum absolute error tolerance. The optimization is
                permitted to round angles with a threshold determined by this
                tolerance.
            extensions: The extensions instance to use when trying to
                cast gates to known types.
        """
        self.extensions = extensions or extension.Extensions()
        self.ignore_failures = ignore_failures
        self.tolerance = tolerance
        self._tol = linalg.Tolerance(atol=tolerance)

    def _rotation_to_clifford_gate(self, pauli: ops.Pauli, half_turns: float
                                   ) -> ops.CliffordGate:
        quarter_turns = round(half_turns * 2) % 4
        if quarter_turns == 1:
            return ops.CliffordGate.from_pauli(pauli, True)
        elif quarter_turns == 2:
            return ops.CliffordGate.from_pauli(pauli)
        elif quarter_turns == 3:
            return ops.CliffordGate.from_pauli(pauli, True).inverse()
        else:
            return ops.CliffordGate.I

    def _matrix_to_clifford_op(self, mat: np.ndarray, qubit: ops.QubitId
                               ) -> Optional[ops.Operation]:
        rotations = decompositions.single_qubit_matrix_to_pauli_rotations(
                                       mat, self.tolerance)
        clifford_gate = ops.CliffordGate.I
        for pauli, half_turns in rotations:
            if self._tol.all_near_zero_mod(half_turns, 0.5):
                clifford_gate = clifford_gate.merged_with(
                    self._rotation_to_clifford_gate(pauli, half_turns))
            else:
                return None
        return clifford_gate(qubit)

    def _convert_one(self, op: ops.Operation) -> ops.OP_TREE:
        # Don't change if it's already a CliffordGate
        if (isinstance(op, ops.GateOperation) and
            isinstance(op.gate, ops.CliffordGate)):
            return op

        # Single qubit gate with known matrix?
        mat = protocols.maybe_unitary_effect(op)
        if mat is not None and len(op.qubits) == 1:
            cliff_op = self._matrix_to_clifford_op(mat, op.qubits[0])
            if cliff_op is not None:
                return cliff_op
            elif self.ignore_failures:
                return op
            else:
                raise ValueError('Single qubit operation is not in the Clifford'
                                 'group: {!r}'.format(op))

        # Provides a decomposition?
        composite_op = self.extensions.try_cast(ops.CompositeOperation, op)
        if composite_op is not None:
            return composite_op.default_decompose()

        # Just let it be?
        if self.ignore_failures:
            return op

        raise TypeError("Don't know how to work with {!r}. "
                        "It isn't a 1-qubit KnownMatrix, "
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
