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

"""Target gateset used for compiling circuits to IonQ device."""

from typing import Any, Dict, Iterator, List, Tuple

import cirq


class IonQTargetGateset(cirq.TwoQubitCompilationTargetGateset):
    """Target gateset for compiling circuits to IonQ devices.

    The gate families accepted by this gateset are:

    Type gate families:
    *  Single-Qubit Gates: `cirq.XPowGate`, `cirq.YPowGate`, `cirq.ZPowGate`.
    *  Two-Qubit Gates: `cirq.XXPowGate`, `cirq.YYPowGate`, `cirq.ZZPowGate`.
    *  Measurement Gate: `cirq.MeasurementGate`.

    Instance gate families:
    *  Single-Qubit Gates: `cirq.H`.
    *  Two-Qubit Gates: `cirq.CNOT`, `cirq.SWAP`.
    """

    def __init__(self, *, atol: float = 1e-8):
        """Initializes CZTargetGateset

        Args:
            atol: A limit on the amount of absolute error introduced by the decomposition.
        """
        super().__init__(
            cirq.H,
            cirq.CNOT,
            cirq.SWAP,
            cirq.XPowGate,
            cirq.YPowGate,
            cirq.ZPowGate,
            cirq.XXPowGate,
            cirq.YYPowGate,
            cirq.ZZPowGate,
            cirq.MeasurementGate,
            cirq.GlobalPhaseGate,
            unroll_circuit_op=False,
        )
        self.atol = atol

    def _decompose_single_qubit_operation(self, op: cirq.Operation, _) -> Iterator[cirq.OP_TREE]:
        qubit = op.qubits[0]
        mat = cirq.unitary(op)
        for gate in cirq.single_qubit_matrix_to_gates(mat, self.atol):
            yield gate(qubit)

    def _decompose_two_qubit_operation(self, op: cirq.Operation, _) -> cirq.OP_TREE:
        if not cirq.has_unitary(op):
            return NotImplemented
        mat = cirq.unitary(op)
        q0, q1 = op.qubits
        naive = cirq.two_qubit_matrix_to_cz_operations(q0, q1, mat, allow_partial_czs=False)
        temp = cirq.map_operations_and_unroll(
            cirq.Circuit(naive),
            lambda op, _: (
                [cirq.H(op.qubits[1]), cirq.CNOT(*op.qubits), cirq.H(op.qubits[1])]
                if op.gate == cirq.CZ
                else op
            ),
        )
        return cirq.merge_k_qubit_unitaries(
            temp, k=1, rewriter=lambda op: self._decompose_single_qubit_operation(op, -1)
        ).all_operations()

    def _decompose_multi_qubit_operation(self, op: cirq.Operation, _) -> cirq.OP_TREE:
        if isinstance(op.gate, cirq.CCZPowGate):
            return decompose_all_to_all_connect_ccz_gate(op.gate, op.qubits)
        return NotImplemented

    @property
    def preprocess_transformers(self) -> List['cirq.TRANSFORMER']:
        """List of transformers which should be run before decomposing individual operations.

        Decompose to three qubit gates because three qubit gates have different decomposition
        for all-to-all connectivity between qubits.
        """
        return [
            cirq.create_transformer_with_kwargs(
                cirq.expand_composite, no_decomp=lambda op: cirq.num_qubits(op) <= 3
            )
        ]

    @property
    def postprocess_transformers(self) -> List['cirq.TRANSFORMER']:
        """List of transformers which should be run after decomposing individual operations."""
        return [cirq.drop_negligible_operations, cirq.drop_empty_moments]

    def __repr__(self) -> str:
        return f'cirq_ionq.IonQTargetGateset(atol={self.atol})'

    def _value_equality_values_(self) -> Any:
        return self.atol

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ['atol'])

    @classmethod
    def _from_json_dict_(cls, atol, **kwargs):
        return cls(atol=atol)


def decompose_all_to_all_connect_ccz_gate(
    ccz_gate: 'cirq.CCZPowGate', qubits: Tuple['cirq.Qid', ...]
) -> 'cirq.OP_TREE':
    """Decomposition of all-to-all connected qubits are different from line qubits or grid qubits.

    For example, for qubits in the same ion trap, the decomposition of CCZ gate will be:

    0: ──────────────@──────────────────@───@───p──────@───
                     │                  │   │          │
    1: ───@──────────┼───────@───p──────┼───X───p^-1───X───
          │          │       │          │
    2: ───X───p^-1───X───p───X───p^-1───X───p──────────────

    where p = T**ccz_gate._exponent
    """
    if len(qubits) != 3:
        raise ValueError(f'Expect 3 qubits for CCZ gate, got {len(qubits)} qubits.')

    a, b, c = qubits

    p = cirq.T**ccz_gate._exponent
    global_phase = 1j ** (2 * ccz_gate.global_shift * ccz_gate._exponent)
    global_phase = (
        complex(global_phase)
        if cirq.is_parameterized(global_phase) and global_phase.is_complex  # type: ignore
        else global_phase
    )
    global_phase_operation = (
        [cirq.global_phase_operation(global_phase)]
        if cirq.is_parameterized(global_phase) or abs(global_phase - 1.0) > 0
        else []
    )

    return global_phase_operation + [
        cirq.CNOT(b, c),
        p(c) ** -1,
        cirq.CNOT(a, c),
        p(c),
        cirq.CNOT(b, c),
        p(c) ** -1,
        cirq.CNOT(a, c),
        p(b),
        p(c),
        cirq.CNOT(a, b),
        p(a),
        p(b) ** -1,
        cirq.CNOT(a, b),
    ]
