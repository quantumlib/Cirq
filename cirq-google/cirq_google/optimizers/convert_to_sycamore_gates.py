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
import math
from typing import List, Optional

import numpy as np

import cirq
from cirq_google.ops import SycamoreGate
from cirq_google.transformers.analytical_decompositions import two_qubit_to_sycamore


@cirq._compat.deprecated_class(
    deadline='v1.0',
    fix='Use cirq.optimize_for_target_gateset and cirq_google.SycamoreTargetGateset instead.',
)
class ConvertToSycamoreGates(cirq.PointOptimizer):
    """Attempts to convert non-native gates into SycamoreGates.

    First, checks if the given operation is already a native sycamore operation.

    Second, checks if the operation has a known unitary. If so, and the gate is a 1-qubit or
    2-qubit gate, then performs circuit synthesis of the operation.

    Third, attempts to `cirq.decompose` to the operation.

    Fourth, if ignore_failures is set, gives up and returns the gate unchanged. Otherwise raises
    a TypeError.
    """

    def __init__(
        self, tabulation: Optional[cirq.TwoQubitGateTabulation] = None, ignore_failures=False
    ) -> None:
        """Inits ConvertToSycamoreGates.

        Args:
            tabulation: If set, a tabulation for the Sycamore gate to use for
                decomposing Matrix gates. If unset, an analytic calculation is
                used for Matrix gates. To get a TwoQubitGateTabulation, call the
                `two_qubit_gate_product_tabulation` method with a base gate (in this case,
                usually cirq_google.SYC) and a maximum infidelity.
            ignore_failures: If set, gates that fail to convert are forwarded
                unchanged. If not set, conversion failures raise a TypeError.

        Raises:
            ValueError: If the tabulation is not a `TwoQubitGateTabulation`.
        """
        super().__init__()
        self.ignore_failures = ignore_failures
        if tabulation is not None and not isinstance(tabulation, cirq.TwoQubitGateTabulation):
            raise ValueError("provided tabulation must be of type cirq.TwoQubitGateTabulation")
        self.tabulation = tabulation

    def _is_native_sycamore_op(self, op: cirq.Operation) -> bool:
        """Check if the given operation is native to a Sycamore device.

        Args:
            op: Input operation.

        Returns:
            True if the operation is native to the gmon, false otherwise.
        """
        gate = op.gate

        if isinstance(
            gate,
            (
                SycamoreGate,
                cirq.MeasurementGate,
                cirq.PhasedXZGate,
                cirq.PhasedXPowGate,
                cirq.XPowGate,
                cirq.YPowGate,
                cirq.ZPowGate,
            ),
        ):
            return True

        if (
            isinstance(gate, cirq.FSimGate)
            and math.isclose(gate.theta, np.pi / 2)
            and math.isclose(gate.phi, np.pi / 6)
        ):
            return True

        if gate is None and isinstance(op.untagged, cirq.CircuitOperation):
            subcircuit = op.untagged.circuit
            return all(self._is_native_sycamore_op(op) for op in subcircuit.all_operations())

        return False

    def _convert_one(self, op: cirq.Operation) -> cirq.OP_TREE:
        """The main conversion step for the PointOptimizer."""
        if not (cirq.has_unitary(op) and 1 <= cirq.num_qubits(op) <= 2):
            return NotImplemented

        if cirq.num_qubits(op) == 1:
            return [*cirq.merge_single_qubit_gates_to_phxz(cirq.Circuit(op)).all_operations()]

        known_decomp = two_qubit_to_sycamore.known_2q_op_to_sycamore_operations(op)
        if known_decomp is not None:
            return known_decomp
        if self.tabulation is not None:
            return two_qubit_to_sycamore._decompose_arbitrary_into_syc_tabulation(
                op, self.tabulation
            )
        return two_qubit_to_sycamore.two_qubit_matrix_to_sycamore_operations(
            op.qubits[0], op.qubits[1], cirq.unitary(op)
        )

    def convert(self, op: cirq.Operation) -> List[cirq.Operation]:
        def on_stuck_raise(bad):
            return TypeError(
                "Don't know how to work with {!r}. "
                "It isn't a native xmon operation, "
                "a 1 or 2 qubit gate with a known unitary, "
                "or composite.".format(bad)
            )

        return cirq.decompose(
            op,
            keep=self._is_native_sycamore_op,
            intercepting_decomposer=self._convert_one,
            on_stuck_raise=None if self.ignore_failures else on_stuck_raise,
            preserve_structure=True,  # keep CircuitOps but decompose their contents
        )

    def optimization_at(
        self, circuit: cirq.Circuit, index: int, op: cirq.Operation
    ) -> Optional[cirq.PointOptimizationSummary]:
        if op.gate is None and not isinstance(op.untagged, cirq.CircuitOperation):
            return None

        # Check for a SWAP and ZZPowGate together
        if isinstance(op.gate, cirq.ZZPowGate) or op.gate == cirq.SWAP:
            op2 = None
            next_index = circuit.next_moment_operating_on(op.qubits, index + 1)
            if next_index is not None:
                ops_in_front = list({circuit.operation_at(q, next_index) for q in op.qubits})
                if len(ops_in_front) == 1 and ops_in_front[0] is not None:
                    op2 = ops_in_front[0]
            else:
                next_index = 0
            if op2 is not None and (
                (op.gate == cirq.SWAP and isinstance(op2.gate, cirq.ZZPowGate))
                or (isinstance(op.gate, cirq.ZZPowGate) and op2.gate == cirq.SWAP)
            ):
                swap_rzz_decomposed = two_qubit_to_sycamore.known_2q_op_to_sycamore_operations(
                    cirq.CircuitOperation(cirq.FrozenCircuit(op, op2))
                )
                assert swap_rzz_decomposed is not None
                return cirq.PointOptimizationSummary(
                    clear_span=next_index - index + 1,
                    clear_qubits=op.qubits,
                    new_operations=swap_rzz_decomposed,
                )

        converted = self.convert(op)
        if len(converted) == 1 and converted[0] is op:
            return None

        return cirq.PointOptimizationSummary(
            clear_span=1, new_operations=converted, clear_qubits=op.qubits
        )
