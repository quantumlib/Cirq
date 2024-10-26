# Copyright 2024 The Cirq Developers
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

"""Target gateset used for compiling circuits to IonQ native gates."""

from types import NotImplementedType
from typing import Any, Dict, Iterator, List, Tuple, Union

import cirq
import numpy as np

from cirq import linalg
from cirq import ops

from cirq_ionq.ionq_native_gates import GPIGate, GPI2Gate, MSGate


class IonqNativeGatesetBase(cirq.TwoQubitCompilationTargetGateset):
    def __init__(self, *gates, atol: float = 1e-8):
        """Base class for IonQ native gate sets

        Args:
            *gates: A list of `cirq.Gate` subclasses / `cirq.Gate` instances /
                `cirq.GateFamily` instances.
            atol: A limit on the amount of absolute error introduced by the decomposition.
        """
        super().__init__(*gates, unroll_circuit_op=False)
        self.atol = atol

    def _decompose_single_qubit_operation(self, op: cirq.Operation, _) -> Iterator[cirq.OP_TREE]:
        qubit = op.qubits[0]
        mat = cirq.unitary(op)
        yield cirq.global_phase_operation(-1j)
        for gate in self.single_qubit_matrix_to_native_gates(mat):
            yield gate(qubit)

    def _decompose_two_qubit_operation(
        self, op: cirq.Operation, _
    ) -> Union[NotImplementedType, cirq.OP_TREE]:
        if not cirq.has_unitary(op):
            return NotImplemented
        mat = cirq.unitary(op)
        q0, q1 = op.qubits
        naive = cirq.two_qubit_matrix_to_cz_operations(
            q0, q1, mat, allow_partial_czs=False, atol=self.atol
        )
        temp = cirq.map_operations_and_unroll(
            cirq.Circuit(naive),
            lambda op, _: (
                [
                    self._hadamard(op.qubits[1])
                    + self._cnot(*op.qubits)
                    + self._hadamard(op.qubits[1])
                ]
                if op.gate == cirq.CZ
                else op
            ),
        )
        return cirq.merge_k_qubit_unitaries(
            temp, k=1, rewriter=lambda op: self._decompose_single_qubit_operation(op, None)
        ).all_operations()

    def _decompose_multi_qubit_operation(
        self, op: cirq.Operation, _
    ) -> Union[NotImplementedType, cirq.OP_TREE]:
        if isinstance(op.gate, cirq.CCZPowGate):
            return self.decompose_all_to_all_connect_ccz_gate(op.gate, op.qubits)
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

    def single_qubit_matrix_to_native_gates(self, mat: np.ndarray) -> List[cirq.Gate]:
        z_rad_before, y_rad, z_rad_after = linalg.deconstruct_single_qubit_matrix_into_angles(mat)
        return [
            GPI2Gate(phi=(np.pi - z_rad_before) / (2.0 * np.pi)),
            GPIGate(phi=(y_rad / 2 + z_rad_after / 2 - z_rad_before / 2) / (2.0 * np.pi)),
            GPI2Gate(phi=(np.pi + z_rad_after) / (2.0 * np.pi)),
        ]

    def _value_equality_values_(self) -> Any:
        return self.atol

    def _value_equality_values_cls_(self) -> Any:
        return type(self)

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ['atol'])

    @classmethod
    def _from_json_dict_(cls, atol, **kwargs):
        return cls(atol=atol)

    def _hadamard(self, qubit):
        return [GPI2Gate(phi=0.25).on(qubit), GPIGate(phi=0).on(qubit)]

    def _cnot(self, *qubits):
        return [
            GPI2Gate(phi=1 / 4).on(qubits[0]),
            MSGate(phi0=0, phi1=0).on(qubits[0], qubits[1]),
            GPI2Gate(phi=1 / 2).on(qubits[1]),
            GPI2Gate(phi=1 / 2).on(qubits[0]),
            GPI2Gate(phi=-1 / 4).on(qubits[0]),
        ]

    def decompose_all_to_all_connect_ccz_gate(
        self, ccz_gate: 'cirq.CCZPowGate', qubits: Tuple['cirq.Qid', ...]
    ) -> 'cirq.OP_TREE':
        """Decomposition of all-to-all connected qubits are different from line
         qubits or grid qubits, ckeckout IonQTargetGateset.

        For example, for qubits in the same ion trap, the decomposition of CCZ
        gate will be:

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
            self._cnot(*[b, c]),
            p(c) ** -1,
            self._cnot(*[a, c]),
            p(c),
            self._cnot(*[b, c]),
            p(c) ** -1,
            self._cnot(*[a, c]),
            p(b),
            p(c),
            self._cnot(*[a, b]),
            p(a),
            p(b) ** -1,
            self._cnot(*[a, b]),
        ]


class AriaNativeGateset(IonqNativeGatesetBase):
    """Target IonQ native gateset for compiling circuits.

    The gates forming this gateset are:
    GPIGate, GPI2Gate, MSGate
    """

    def __init__(self, *, atol: float = 1e-8):
        """Initializes AriaNativeGateset

        Args:
            atol: A limit on the amount of absolute error introduced by the decomposition.
        """
        super().__init__(GPIGate, GPI2Gate, MSGate, ops.MeasurementGate, atol=atol)

    def __repr__(self) -> str:
        return f'cirq_ionq.AriaNativeGateset(atol={self.atol})'


class ForteNativeGateset(IonqNativeGatesetBase):
    """Target IonQ native gateset for compiling circuits.

    The gates forming this gateset are:
    GPIGate, GPI2Gate, MSGate
    Note: in the future ZZGate might be added here.
    """

    def __init__(self, *, atol: float = 1e-8):
        """Initializes ForteNativeGateset

        Args:
            atol: A limit on the amount of absolute error introduced by the decomposition.
        """
        super().__init__(GPIGate, GPI2Gate, MSGate, ops.MeasurementGate, atol=atol)

    def __repr__(self) -> str:
        return f'cirq_ionq.ForteNativeGateset(atol={self.atol})'
