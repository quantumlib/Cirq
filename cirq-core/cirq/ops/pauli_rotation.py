# Copyright 2025 The Cirq Developers
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

from collections.abc import Sequence
from types import NotImplementedType
from typing import TYPE_CHECKING

import numpy as np

from cirq import protocols, value
from cirq.ops import dense_pauli_string as dps, gate_operation, pauli_string_phasor, raw_types

if TYPE_CHECKING:
    import cirq


@value.value_equality(approximate=True)
class PauliRotationGate(raw_types.Gate):
    r"""A gate representing :math:`e^{i \theta P}` for a Pauli string :math:`P`.

    The Pauli string is specified as a `cirq.DensePauliString`, which preserves
    identity factors (unlike `cirq.PauliString`). For a unit Pauli operator
    :math:`P`, the unitary is :math:`\cos(\theta) I + i \sin(\theta) P`.
    """

    def __init__(
        self, dense_pauli_string: dps.DensePauliString, *, exponent: cirq.TParamVal
    ) -> None:
        if dense_pauli_string.coefficient != 1:
            raise ValueError(
                'PauliRotationGate requires a unit Pauli string with coefficient 1.'
            )
        self._dense_pauli_string = dense_pauli_string
        self._exponent = exponent

    @property
    def dense_pauli_string(self) -> dps.DensePauliString:
        return self._dense_pauli_string

    @property
    def exponent(self) -> cirq.TParamVal:
        return self._exponent

    def _value_equality_values_(self):
        return self._dense_pauli_string, self._exponent

    def num_qubits(self) -> int:
        return len(self._dense_pauli_string.pauli_mask)

    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self._exponent)

    def _parameter_names_(self) -> tuple[str, ...]:
        return protocols.parameter_names(self._exponent)

    def _resolve_parameters_(
        self, resolver: cirq.ParamResolver, recursive: bool
    ) -> PauliRotationGate:
        return PauliRotationGate(
            self._dense_pauli_string,
            exponent=protocols.resolve_parameters(self._exponent, resolver, recursive),
        )

    def _unitary_(self) -> np.ndarray | NotImplementedType:
        if protocols.is_parameterized(self._exponent):
            return NotImplemented
        pauli_unitary = protocols.unitary(self._dense_pauli_string)
        cos_theta = np.cos(self._exponent)
        sin_theta = np.sin(self._exponent)
        identity = np.eye(pauli_unitary.shape[0], dtype=complex)
        return cos_theta * identity + 1j * sin_theta * pauli_unitary

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        if protocols.is_parameterized(self._exponent):
            return NotImplemented
        # Pass explicit qubits so PauliStringPhasor keeps identity factors in the
        # full Hilbert space (e.g. X⊗I rather than a single-qubit X).
        pauli_op = self._dense_pauli_string.on(*qubits)
        pauli_string = (
            pauli_op.gate
            if isinstance(pauli_op, gate_operation.GateOperation)
            else pauli_op
        )
        return [
            pauli_string_phasor.PauliStringPhasor(
                pauli_string,
                qubits=qubits,
                exponent_pos=self._exponent / np.pi,
                exponent_neg=-self._exponent / np.pi,
            )
        ]

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> str:
        return f'PR({self._dense_pauli_string},{self._exponent})'

    def __pow__(self, power: int) -> PauliRotationGate:
        return PauliRotationGate(self._dense_pauli_string, exponent=self._exponent * power)

    def __repr__(self) -> str:
        return (
            f'cirq.PauliRotationGate({self._dense_pauli_string!r}, '
            f'exponent={self._exponent!r})'
        )


@value.value_equality(approximate=True)
class PauliRotation(gate_operation.GateOperation):
    r"""An operation representing :math:`e^{i \theta P}` for a Pauli string :math:`P`.

    Accepts a `cirq.DensePauliString` label or string such as ``'XI'``.
    """

    def __init__(
        self,
        pauli_string: dps.DensePauliString | str,
        qubits: Sequence[cirq.Qid],
        *,
        exponent: cirq.TParamVal,
    ) -> None:
        if isinstance(pauli_string, str):
            dense_pauli_string = dps.DensePauliString(pauli_string)
        else:
            dense_pauli_string = pauli_string
        if len(dense_pauli_string.pauli_mask) != len(qubits):
            raise ValueError(
                'Pauli string length must match number of qubits. '
                f'Got {len(dense_pauli_string.pauli_mask)} Paulis and {len(qubits)} qubits.'
            )
        gate = PauliRotationGate(dense_pauli_string, exponent=exponent)
        super().__init__(gate, qubits)

    @property
    def gate(self) -> PauliRotationGate:
        return super().gate  # type: ignore[return-value]

    @property
    def exponent(self) -> cirq.TParamVal:
        return self.gate.exponent

    @property
    def dense_pauli_string(self) -> dps.DensePauliString:
        return self.gate.dense_pauli_string

    def _value_equality_values_(self):
        return self.gate, self.qubits

    def __pow__(self, power: int) -> PauliRotation:
        return PauliRotation(
            self.dense_pauli_string, self.qubits, exponent=self.exponent * power
        )

    def __repr__(self) -> str:
        return (
            f'cirq.PauliRotation({self.dense_pauli_string!r}, '
            f'{self.qubits!r}, exponent={self.exponent!r})'
        )
