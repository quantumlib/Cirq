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
#
from typing import Any, Dict, List, Type, Union

import cirq
from cirq.protocols.decompose_protocol import DecomposeResult


class PasqalGateset(cirq.CompilationTargetGateset):
    """A Compilation target intended for Pasqal neutral atom devices.
    This gateset supports single qubit gates that can be used
    in a parallel fashion as well as CZ.

    This gateset can optionally include CNOT, CCNOT (TOFFOLI) gates, and
    CCZ as well.

    Args:
        include_additional_controlled_ops: Whether to include CCZ, CCNOT, and CNOT
            gates (defaults to True).
    """

    def __init__(self, include_additional_controlled_ops: bool = True):
        gate_families: List[Union[Type[cirq.Gate], cirq.Gate, cirq.GateFamily]] = [
            cirq.ParallelGateFamily(cirq.H),
            cirq.ParallelGateFamily(cirq.PhasedXPowGate),
            cirq.ParallelGateFamily(cirq.XPowGate),
            cirq.ParallelGateFamily(cirq.YPowGate),
            cirq.ParallelGateFamily(cirq.ZPowGate),
            cirq.AnyIntegerPowerGateFamily(cirq.CZPowGate),
            cirq.IdentityGate,
            cirq.MeasurementGate,
        ]
        self.include_additional_controlled_ops = include_additional_controlled_ops
        if self.include_additional_controlled_ops:
            gate_families.append(cirq.AnyIntegerPowerGateFamily(cirq.CNotPowGate))
            gate_families.append(cirq.AnyIntegerPowerGateFamily(cirq.CCNotPowGate))
            gate_families.append(cirq.AnyIntegerPowerGateFamily(cirq.CCZPowGate))

        super().__init__(*gate_families, unroll_circuit_op=False)

    @property
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
        mat = cirq.unitary(op, None) if len(op.qubits) <= 2 else None
        if mat is not None and len(op.qubits) == 1:
            gates = cirq.single_qubit_matrix_to_phased_x_z(mat)
            return [g.on(op.qubits[0]) for g in gates]
        if mat is not None and len(op.qubits) == 2:
            return cirq.two_qubit_matrix_to_cz_operations(
                op.qubits[0], op.qubits[1], mat, allow_partial_czs=False, clean_operations=True
            )

        return NotImplemented

    @property
    def preprocess_transformers(self) -> List['cirq.TRANSFORMER']:
        return []

    @property
    def postprocess_transformers(self) -> List['cirq.TRANSFORMER']:
        return []

    def __repr__(self):
        return (
            f'cirq_pasqal.PasqalGateset(include_additional_controlled_ops='
            f'{self.include_additional_controlled_ops})'
        )

    @classmethod
    def _from_json_dict_(cls, include_additional_controlled_ops, **kwargs):
        return cls(include_additional_controlled_ops=include_additional_controlled_ops)

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.protocols.obj_to_dict_helper(self, ['include_additional_controlled_ops'])
