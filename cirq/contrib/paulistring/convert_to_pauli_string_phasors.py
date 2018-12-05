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

from typing import Optional, cast, TYPE_CHECKING

import numpy as np

from cirq import ops, linalg, optimizers, protocols
from cirq.circuits.circuit import Circuit
from cirq.circuits.optimization_pass import (
    PointOptimizationSummary,
    PointOptimizer,
)
from cirq.contrib.paulistring.pauli_string_phasor import PauliStringPhasor

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import List


class ConvertToPauliStringPhasors(PointOptimizer):
    """Attempts to convert single-qubit gates into single-qubit
    PauliStringPhasor operations.

    Checks if the operation has a known unitary effect. If so, and the gate is a
        1-qubit gate, then decomposes it into x, y, or z rotations and creates a
        PauliStringPhasor for each.
    """

    def __init__(self,
                 ignore_failures: bool = False,
                 keep_clifford: bool = False,
                 tolerance: float = 0) -> None:
        """
        Args:
            ignore_failures: If set, gates that fail to convert are forwarded
                unchanged. If not set, conversion failures raise a TypeError.
            keep_clifford: If set, single qubit rotations in the Clifford group
                are converted to SingleQubitCliffordGates.
            tolerance: Maximum absolute error tolerance. The optimization is
                permitted to round angles with a threshold determined by this
                tolerance.
        """
        super().__init__()
        self.ignore_failures = ignore_failures
        self.keep_clifford = keep_clifford
        self.tolerance = tolerance
        self._tol = linalg.Tolerance(atol=tolerance)

    def _matrix_to_pauli_string_phasors(self,
                                        mat: np.ndarray,
                                        qubit: ops.QubitId) -> ops.OP_TREE:
        rotations = optimizers.single_qubit_matrix_to_pauli_rotations(
            mat, self.tolerance)
        out_ops = []  # type: List[ops.Operation]
        for pauli, half_turns in rotations:
            if (self.keep_clifford
                    and self._tol.all_near_zero_mod(half_turns, 0.5)):
                cliff_gate = ops.SingleQubitCliffordGate.from_quarter_turns(
                    pauli, round(half_turns * 2))
                if out_ops and not isinstance(out_ops[-1], PauliStringPhasor):
                    op = cast(ops.GateOperation, out_ops[-1])
                    gate = cast(ops.SingleQubitCliffordGate, op.gate)
                    out_ops[-1] = gate.merged_with(cliff_gate)(qubit)
                else:
                    out_ops.append(
                        cliff_gate(qubit))
            else:
                pauli_string = ops.PauliString.from_single(qubit, pauli)
                out_ops.append(
                    PauliStringPhasor(pauli_string,
                                      half_turns=round(half_turns, 10)))
        return out_ops

    def _convert_one(self, op: ops.Operation) -> ops.OP_TREE:
        # Don't change if it's already a PauliStringPhasor
        if isinstance(op, PauliStringPhasor):
            return op

        if (self.keep_clifford
            and isinstance(op, ops.GateOperation)
                and isinstance(op.gate, ops.SingleQubitCliffordGate)):
            return op

        # Single qubit gate with known matrix?
        if len(op.qubits) == 1:
            mat = protocols.unitary(op, None)
            if mat is not None:
                return self._matrix_to_pauli_string_phasors(mat, op.qubits[0])

        # Just let it be?
        if self.ignore_failures:
            return op

        raise TypeError("Don't know how to work with {!r}. "
                        "It isn't a 1-qubit operation with a known unitary "
                        "effect.".format(op))

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
