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
from typing import List

from cirq import ops, protocols
from cirq.circuits.optimization_pass import (
    PointOptimizationSummary,
    PointOptimizer,
)
from cirq import optimizers


class ConvertToNeutralAtomGates(PointOptimizer):
    """Attempts to convert gates into native Atom gates.

    First, checks if the given operation is already a native neutral atom
    operation.

    Second, checks if the operation has a known unitary. If so, and the gate
        is a 1-qubit or 2-qubit gate, then performs circuit synthesis of the
        operation. The 2-qubit gates are decomposed using CZ gates because
        CZ gates are the highest fidelity 2-qubit gates for neutral atoms.

    Third, attempts to `cirq.decompose` to the operation.

    Fourth, if ignore_failures is set, gives up and returns the gate unchanged.
        Otherwise raises a TypeError.
    """

    def __init__(self, ignore_failures=False) -> None:
        """
        Args:
            ignore_failures: If set, gates that fail to convert are forwarded
                unchanged. If not set, conversion failures raise a TypeError.
        """
        super().__init__()
        self.ignore_failures = ignore_failures

    def _convert_one(self, op: ops.Operation) -> ops.OP_TREE:
        # Known matrix?
        mat = protocols.unitary(op, None) if len(op.qubits) <= 2 else None
        if mat is not None and len(op.qubits) == 1:
            gates = optimizers.single_qubit_matrix_to_phased_x_z(mat)
            return [g.on(op.qubits[0]) for g in gates]
        if mat is not None and len(op.qubits) == 2:
            return optimizers.two_qubit_matrix_to_operations(
                op.qubits[0], op.qubits[1], mat, allow_partial_czs=False, clean_operations=True
            )

        return NotImplemented

    def convert(self, op: ops.Operation) -> List[ops.Operation]:
        def on_stuck_raise(bad):
            return TypeError(
                "Don't know how to work with {!r}. "
                "It isn't a native atom operation, "
                "a 1 or 2 qubit gate with a known unitary, "
                "or composite.".format(bad)
            )

        return protocols.decompose(
            op,
            keep=is_native_neutral_atom_op,
            intercepting_decomposer=self._convert_one,
            on_stuck_raise=None if self.ignore_failures else on_stuck_raise,
        )

    def optimization_at(self, circuit, index, op):
        converted = self.convert(op)
        if len(converted) == 1 and converted[0] is op:
            return None
        return PointOptimizationSummary(
            clear_span=1, new_operations=converted, clear_qubits=op.qubits
        )


def is_native_neutral_atom_op(operation: ops.Operation) -> bool:
    if isinstance(operation, (ops.GateOperation, ops.ParallelGateOperation)):
        return is_native_neutral_atom_gate(operation.gate)
    return False


def is_native_neutral_atom_gate(gate: ops.Gate) -> bool:
    if not isinstance(
        gate,
        (
            ops.CCXPowGate,
            ops.CCZPowGate,
            ops.CZPowGate,
            ops.CNotPowGate,
            ops.XPowGate,
            ops.YPowGate,
            ops.PhasedXPowGate,
            ops.MeasurementGate,
            ops.ZPowGate,
            ops.IdentityGate,
        ),
    ):
        return False
    if isinstance(gate, (ops.CNotPowGate, ops.CZPowGate, ops.CCXPowGate, ops.CCZPowGate)):
        if not gate.exponent.is_integer():
            return False
    return True
