# Copyright 2022 The Cirq Developers
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
from typing import List, Optional, TYPE_CHECKING

from cirq import ops, protocols, transformers
from cirq.protocols.decompose_protocol import DecomposeResult


if TYPE_CHECKING:
    import cirq


class NeutralAtomGateset(transformers.CompilationTargetGateset):
    """A Compilation target intended for neutral atom devices.

    This gateset supports CNOT, CCNOT (TOFFOLI) gates, CZ,
    CCZ gates, as well as single qubits gates that can be used
    in a parallel fashion.  The maximum amount of parallelism
    can be set by arguments.

    This compilation gateset decomposes operations into CZ
    because CZ gates are the highest fidelity two qubit gates
    for neutral atoms.

    Args:
        max_parallel_z: The maximum amount of parallelism for
            Z gates.
        max_parallel_xy: The maximum amount of parallelism for
            X, Y and PhasedXPow gates.
    """

    def __init__(self, max_parallel_z: Optional[int] = None, max_parallel_xy: Optional[int] = None):
        super().__init__(
            ops.AnyIntegerPowerGateFamily(ops.CNotPowGate),
            ops.AnyIntegerPowerGateFamily(ops.CCNotPowGate),
            ops.AnyIntegerPowerGateFamily(ops.CZPowGate),
            ops.AnyIntegerPowerGateFamily(ops.CCZPowGate),
            ops.ParallelGateFamily(ops.ZPowGate, max_parallel_allowed=max_parallel_z),
            ops.ParallelGateFamily(ops.XPowGate, max_parallel_allowed=max_parallel_xy),
            ops.ParallelGateFamily(ops.YPowGate, max_parallel_allowed=max_parallel_xy),
            ops.ParallelGateFamily(ops.PhasedXPowGate, max_parallel_allowed=max_parallel_xy),
            ops.MeasurementGate,
            ops.IdentityGate,
            unroll_circuit_op=False,
        )

    def num_qubits(self) -> int:
        """Maximum number of qubits on which a gate from this gateset can act upon."""
        return 2

    def decompose_to_target_gateset(self, op: 'cirq.Operation', moment_idx: int) -> DecomposeResult:
        """Method to rewrite the given operation using gates from this gateset.
        Args:
            op: `cirq.Operation` to be rewritten using gates from this gateset.
            moment_idx: Moment index where the given operation `op` occurs in a circuit.
        Returns:
            - An equivalent `cirq.OP_TREE` implementing `op` using gates from this gateset.
            - `None` or `NotImplemented` if does not know how to decompose `op`.
        """
        # Known matrix?
        mat = protocols.unitary(op, None) if len(op.qubits) <= 2 else None
        if mat is not None and len(op.qubits) == 1:
            gates = transformers.single_qubit_matrix_to_phased_x_z(mat)
            return [g.on(op.qubits[0]) for g in gates]
        if mat is not None and len(op.qubits) == 2:
            return transformers.two_qubit_matrix_to_cz_operations(
                op.qubits[0], op.qubits[1], mat, allow_partial_czs=False, clean_operations=True
            )

        return NotImplemented

    @property
    def preprocess_transformers(self) -> List['cirq.TRANSFORMER']:
        return []

    @property
    def postprocess_transformers(self) -> List['cirq.TRANSFORMER']:
        return []
