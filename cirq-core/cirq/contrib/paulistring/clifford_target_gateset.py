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

from __future__ import annotations

from enum import Enum
from types import NotImplementedType
from typing import cast, List, Type, TYPE_CHECKING, Union

import numpy as np

from cirq import linalg, ops, protocols, transformers

if TYPE_CHECKING:
    import cirq


def _matrix_to_clifford_op(
    mat: np.ndarray, qubit: cirq.Qid, *, atol: float
) -> Union[ops.Operation, NotImplementedType]:
    rotations = transformers.single_qubit_matrix_to_pauli_rotations(mat, atol)
    clifford_gate = ops.SingleQubitCliffordGate.I
    for pauli, half_turns in rotations:
        if linalg.all_near_zero_mod(half_turns, 0.5):
            quarter_turns = round(half_turns * 2) % 4
            # quarter_turns will always be 1-sqrt(pauli) / 2-pauli / 3-sqrt(pauli) ** -1.
            clifford_gate = clifford_gate.merged_with(
                ops.SingleQubitCliffordGate.from_pauli(pauli, sqrt=bool(quarter_turns % 2))
                ** (1 - 2 * int(quarter_turns == 3))
            )
        else:
            return NotImplemented
    return clifford_gate(qubit)


def _matrix_to_pauli_string_phasors(
    mat: np.ndarray, qubit: cirq.Qid, *, keep_clifford: bool, atol: float
) -> ops.OP_TREE:
    rotations = transformers.single_qubit_matrix_to_pauli_rotations(mat, atol)
    out_ops: List[ops.GateOperation] = []
    for pauli, half_turns in rotations:
        if keep_clifford and linalg.all_near_zero_mod(half_turns, 0.5):
            cliff_gate = ops.SingleQubitCliffordGate.from_quarter_turns(
                pauli, round(half_turns * 2)
            )
            if out_ops and not isinstance(out_ops[-1], ops.PauliStringPhasor):
                gate = cast(ops.SingleQubitCliffordGate, out_ops[-1].gate)
                out_ops[-1] = gate.merged_with(cliff_gate)(qubit)
            else:
                out_ops.append(cliff_gate(qubit))
        else:
            out_ops.append(
                ops.PauliStringPhasor(
                    ops.PauliString(pauli.on(qubit)), exponent_neg=round(half_turns, 10)
                )
            )
    return out_ops


class CliffordTargetGateset(transformers.TwoQubitCompilationTargetGateset):
    """Target gateset containing CZ + Meas + SingleQubitClifford / PauliStringPhasor gates."""

    class SingleQubitTarget(Enum):
        SINGLE_QUBIT_CLIFFORDS = 1
        PAULI_STRING_PHASORS_AND_CLIFFORDS = 2
        PAULI_STRING_PHASORS = 3

    def __init__(
        self,
        *,
        single_qubit_target: SingleQubitTarget = SingleQubitTarget.PAULI_STRING_PHASORS_AND_CLIFFORDS,  # pylint: disable=line-too-long
        atol: float = 1e-8,
    ):
        """Initializes CliffordTargetGateset

        Args:
            single_qubit_target: Specifies the decomposition strategy for single qubit gates.
                SINGLE_QUBIT_CLIFFORDS: Decompose all single qubit gates to
                    `cirq.SingleQubitCliffordGate`.
                PAULI_STRING_PHASORS_AND_CLIFFORDS: Accept both `cirq.SingleQubitCliffordGate` and
                    `cirq.PauliStringPhasorGate`; but decompose unknown gates into
                    `cirq.PauliStringPhasorGate`.
                PAULI_STRING_PHASORS: Decompose all single qubit gates to
                    `cirq.PauliStringPhasorGate`.
            atol: A limit on the amount of absolute error introduced by the decomposition.
        """
        self.atol = atol
        self.single_qubit_target = single_qubit_target
        gates: List[Union[cirq.Gate, Type[cirq.Gate]]] = [ops.CZ, ops.MeasurementGate]
        if single_qubit_target in [
            self.SingleQubitTarget.SINGLE_QUBIT_CLIFFORDS,
            self.SingleQubitTarget.PAULI_STRING_PHASORS_AND_CLIFFORDS,
        ]:
            gates.append(ops.SingleQubitCliffordGate)
        if single_qubit_target in [
            self.SingleQubitTarget.PAULI_STRING_PHASORS_AND_CLIFFORDS,
            self.SingleQubitTarget.PAULI_STRING_PHASORS,
        ]:
            gates.append(ops.PauliStringPhasorGate)
        super().__init__(*gates)

    def _decompose_single_qubit_operation(
        self, op: cirq.Operation, _
    ) -> Union[NotImplementedType, cirq.OP_TREE]:
        if not protocols.has_unitary(op):
            return NotImplemented
        mat = protocols.unitary(op)
        keep_clifford = (
            self.single_qubit_target == self.SingleQubitTarget.PAULI_STRING_PHASORS_AND_CLIFFORDS
        )
        return (
            _matrix_to_clifford_op(mat, op.qubits[0], atol=self.atol)
            if self.single_qubit_target == self.SingleQubitTarget.SINGLE_QUBIT_CLIFFORDS
            else _matrix_to_pauli_string_phasors(
                mat, op.qubits[0], keep_clifford=keep_clifford, atol=self.atol
            )
        )

    def _decompose_two_qubit_operation(
        self, op: cirq.Operation, _
    ) -> Union[NotImplementedType, cirq.OP_TREE]:
        if not protocols.has_unitary(op):
            return NotImplemented
        return transformers.two_qubit_matrix_to_cz_operations(
            op.qubits[0],
            op.qubits[1],
            protocols.unitary(op),
            allow_partial_czs=False,
            atol=self.atol,
        )

    @property
    def postprocess_transformers(self) -> List[cirq.TRANSFORMER]:
        """List of transformers which should be run after decomposing individual operations."""

        def rewriter(o: cirq.CircuitOperation):
            result = self._decompose_single_qubit_operation(o, -1)
            return o.circuit.all_operations() if result is NotImplemented else result

        return [
            transformers.create_transformer_with_kwargs(
                transformers.merge_k_qubit_unitaries, k=1, rewriter=rewriter
            ),
            transformers.drop_negligible_operations,
            transformers.drop_empty_moments,
        ]
