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

from cirq import ops, decompositions, protocols
from cirq.circuits.circuit import Circuit
from cirq.circuits.optimization_pass import (
    PointOptimizationSummary,
    PointOptimizer,
)


class ConvertToCzAndSingleGates(PointOptimizer):
    """Attempts to convert strange multi-qubit gates into CZ and single qubit
    gates.

    First, checks if the operation has a unitary effect. If so, and the gate is
        a 1-qubit or 2-qubit gate, then performs circuit synthesis of the
        operation.

    Second, attempts to `cirq.decompose` to the operation.

    Third, if ignore_failures is set, gives up and returns the gate unchanged.
        Otherwise raises a TypeError.
    """

    def __init__(self,
                 ignore_failures: bool = False,
                 allow_partial_czs: bool = False) -> None:
        """
        Args:
            ignore_failures: If set, gates that fail to convert are forwarded
                unchanged. If not set, conversion failures raise a TypeError.
            allow_partial_czs: If set, the decomposition is permitted to use
                gates of the form `cirq.CZ**t`, instead of only `cirq.CZ`.
        """
        super().__init__()
        self.ignore_failures = ignore_failures
        self.allow_partial_czs = allow_partial_czs

    def convert(self, op: ops.Operation) -> ops.OP_TREE:
        def keep(op: ops.Operation) -> bool:
            return (# Check if this is a CZ
                    # Only keep partial CZ gates if allow_partial_czs
                    (isinstance(op, ops.GateOperation)
                     and isinstance(op.gate, ops.CZPowGate)
                     and (self.allow_partial_czs or op.gate.exponent == 1))
                    # Measurement?
                    or ops.MeasurementGate.is_measurement(op)
                    # SingleQubit known matrix
                    or (protocols.unitary(op, None) is not None
                        and len(op.qubits) == 1))

        def convert_one(op: ops.Operation) -> ops.OP_TREE:
            # Known matrix?
            mat = protocols.unitary(op, None)
            if mat is not None and len(op.qubits) == 2:
                return decompositions.two_qubit_matrix_to_operations(
                    op.qubits[0],
                    op.qubits[1],
                    mat,
                    allow_partial_czs=False)
            return NotImplemented

        def on_stuck_raise(op: ops.Operation):
            raise TypeError("Don't know how to work with {!r}. "
                            "It isn't composite or an operation with a "
                            "known unitary effect on 1 or 2 qubits.".format(op))

        return protocols.decompose(op, intercepting_decomposer=convert_one,
                                   keep=keep,
                                   on_stuck_raise=(None if self.ignore_failures
                                                   else on_stuck_raise))

    def optimization_at(self, circuit: Circuit, index: int, op: ops.Operation
                        ) -> Optional[PointOptimizationSummary]:
        converted = self.convert(op)
        if converted is op:
            return None

        return PointOptimizationSummary(
            clear_span=1,
            new_operations=converted,
            clear_qubits=op.qubits)
