# Copyright 2019 The Cirq Developers
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
from typing import List, cast

import numpy as np

from cirq import circuits, ops, protocols, optimizers
from cirq.google import SycamoreGate
from cirq.google.optimizers.textbook_gates_from_sycamore import (
    known_two_q_operations_to_sycamore_operations, swap_zztheta)


class ConvertToSycamoreGates(circuits.PointOptimizer):
    """Attempts to convert non-native gates into SycamoreGates.

    First, checks if the given operation is already a native sycamore operation.

    Second, checks if the operation has a known unitary. If so, and the gate
        is a 1-qubit or 2-qubit gate, then performs circuit synthesis of the
        operation.

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

    def _is_native_sycamore_op(self, op: ops.Operation) -> bool:
        """Check if the given operation is native to a Sycamore device.

        Args:
            op: Input operation.

        Returns:
            True if the operation is native to the gmon, false otherwise.
        """
        return (isinstance(op, ops.GateOperation) and isinstance(
            cast(ops.GateOperation, op).gate,
            (SycamoreGate, ops.MeasurementGate, ops.PhasedXPowGate,
             ops.XPowGate, ops.YPowGate, ops.ZPowGate)))

    def _convert_one(self, op: ops.Operation) -> ops.OP_TREE:
        """
        Decomposer intercept:  Upon cirq.protocols.decompose catch and
        return new OP_Tree

        This should decompose based on number of qubits.
        """
        if len(op.qubits) == 1:
            mat = protocols.unitary(op, None)
            gates = optimizers.single_qubit_matrix_to_phased_x_z(mat)
            return [g.on(op.qubits[0]) for g in gates]
        elif len(op.qubits) == 2 and isinstance(op, ops.GateOperation):
            return known_two_q_operations_to_sycamore_operations(
                op.qubits[0], op.qubits[1], op)

        return NotImplemented

    def convert(self, op: ops.Operation) -> List[ops.Operation]:

        def on_stuck_raise(bad):
            return TypeError("Don't know how to work with {!r}. "
                             "It isn't a native xmon operation, "
                             "a 1 or 2 qubit gate with a known unitary, "
                             "or composite.".format(bad))

        return protocols.decompose(
            op,
            keep=self._is_native_sycamore_op,
            intercepting_decomposer=self._convert_one,
            on_stuck_raise=None if self.ignore_failures else on_stuck_raise)

    def optimization_at(self, circuit, index, op):

        if not isinstance(op, ops.GateOperation):
            return None

        gate = op.gate

        # Check for a SWAP and ZZPowGate together
        if isinstance(gate, ops.ZZPowGate) or gate == ops.SWAP:
            gate2 = None
            rads = None
            next_index = circuit.next_moment_operating_on(op.qubits, index + 1)
            if next_index is not None:
                ops_in_front = list(
                    {circuit.operation_at(q, next_index) for q in op.qubits})
                if len(ops_in_front) == 1 and isinstance(
                        ops_in_front[0], ops.GateOperation):
                    gate2 = ops_in_front[0].gate

            if (isinstance(gate, ops.SwapPowGate) and
                    isinstance(gate2, ops.ZZPowGate)):
                rads = gate2.exponent * np.pi / 2
            if (isinstance(gate, ops.ZZPowGate) and gate2 == ops.SWAP):
                rads = gate.exponent * np.pi / 2
            if rads is not None:
                return circuits.PointOptimizationSummary(
                    clear_span=next_index - index + 1,
                    clear_qubits=op.qubits,
                    new_operations=swap_zztheta(rads, op.qubits[0],
                                                op.qubits[1]))

        converted = self.convert(op)
        if len(converted) == 1 and converted[0] is op:
            return None

        return circuits.PointOptimizationSummary(clear_span=1,
                                                 new_operations=converted,
                                                 clear_qubits=op.qubits)
