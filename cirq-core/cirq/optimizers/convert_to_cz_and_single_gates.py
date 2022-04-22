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

from cirq import circuits, ops, protocols, _compat
from cirq.transformers.analytical_decompositions import two_qubit_to_cz


@_compat.deprecated_class(
    deadline='v1.0', fix='Use cirq.optimize_for_target_gateset and cirq.CZTargetGateset instead.'
)
class ConvertToCzAndSingleGates(circuits.PointOptimizer):
    """Attempts to convert strange multi-qubit gates into CZ and single qubit
    gates.

    First, checks if the operation has a unitary effect. If so, and the gate is
        a 1-qubit or 2-qubit gate, then performs circuit synthesis of the
        operation.

    Second, attempts to `cirq.decompose` to the operation.

    Third, if ignore_failures is set, gives up and returns the gate unchanged.
        Otherwise raises a TypeError.
    """

    def __init__(self, ignore_failures: bool = False, allow_partial_czs: bool = False) -> None:
        """Inits ConvertToCzAndSingleGates.

        Args:
            ignore_failures: If set, gates that fail to convert are forwarded
                unchanged. If not set, conversion failures raise a TypeError.
            allow_partial_czs: If set, the decomposition is permitted to use
                gates of the form `cirq.CZ**t`, instead of only `cirq.CZ`.
        """
        super().__init__()
        self.ignore_failures = ignore_failures
        self.allow_partial_czs = allow_partial_czs
        self.gateset = ops.Gateset(
            ops.CZPowGate if allow_partial_czs else ops.CZ,
            ops.MeasurementGate,
            ops.AnyUnitaryGateFamily(1),
            ops.GlobalPhaseGate,
        )

    def _decompose_two_qubit_unitaries(self, op: ops.Operation) -> ops.OP_TREE:
        # Known matrix?
        if len(op.qubits) == 2:
            mat = protocols.unitary(op, None)
            if mat is not None:
                return two_qubit_to_cz.two_qubit_matrix_to_cz_operations(
                    op.qubits[0], op.qubits[1], mat, allow_partial_czs=self.allow_partial_czs
                )
        return NotImplemented

    def _on_stuck_raise(self, op: ops.Operation):
        raise TypeError(
            "Don't know how to work with {!r}. "
            "It isn't composite or an operation with a "
            "known unitary effect on 1 or 2 qubits.".format(op)
        )

    def optimization_at(
        self, circuit: circuits.Circuit, index: int, op: ops.Operation
    ) -> Optional[circuits.PointOptimizationSummary]:
        converted = protocols.decompose(
            op,
            intercepting_decomposer=self._decompose_two_qubit_unitaries,
            keep=self.gateset._validate_operation,
            on_stuck_raise=(None if self.ignore_failures else self._on_stuck_raise),
        )
        if converted == [op]:
            return None

        return circuits.PointOptimizationSummary(
            clear_span=1, new_operations=converted, clear_qubits=op.qubits
        )
